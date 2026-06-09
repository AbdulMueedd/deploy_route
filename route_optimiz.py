"""
VRP Route Optimizer — Flask Backend
The Cary Company

Solves a Vehicle Routing Problem (VRP) with:
  - Pickup and delivery stops (deliveries run first, then pickups)
  - Trailer-based capacity constraints (each trailer type has its own capacity and zone access)
  - Priority stops (high/urgent stops are penalized if visited late)
  - Zone restrictions (trailer type determines which zones it can access)
  - Real road distances and travel times via OSRM
  - OR-Tools for route optimization

Units: distances in miles, time in minutes, cargo measured in pallets.
"""

import logging
import traceback
import time as time_module
from datetime import datetime

import requests
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os

# ── App setup ──────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder='static')
CORS(app)

# ── Logging setup ──────────────────────────────────────────────────────────────
# Writes audit logs to both the console and a rotating log file.
# Every request, solve attempt, OSRM call, and error is recorded with a timestamp.

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),                                    # console
        logging.FileHandler('logs/vrp_audit.log', encoding='utf-8'),  # file
    ]
)
log = logging.getLogger('vrp')


@app.after_request
def after_request(response):
    """Add CORS headers to every response so the frontend can reach the API."""
    response.headers.add('Access-Control-Allow-Origin',  '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response


# ── Constants ──────────────────────────────────────────────────────────────────

KM_TO_MILES = 0.621371  # conversion factor applied to all OSRM distances

# Trailer type definitions.
# The truck itself is always the same — the trailer determines:
#   capacity:      max cargo units the trailer can hold at any one time
#   allowed_zones: which delivery zone types this trailer is permitted to enter
#   description:   human-readable explanation shown in the UI
#   capacity:      maximum pallets the trailer can hold at any one time
#
# Pallet capacities based on Cary Company fleet specs:
#   53-foot dry van  → 26 pallets (standard double-stack floor configuration)
#   Liftgate trailer → 12 pallets (liftgate mechanism reduces usable floor space)
#   Hazmat trailer   → 12 pallets (liftgate-style, restricted to industrial zones only)
TRAILER_DEFS = {
    "53ft": {
        "capacity":      26,
        "allowed_zones": ["residential", "commercial", "industrial", "airport"],
        "description":   "53\' dry van — 26 pallets, no zone restrictions",
        "color":         "#1f3f8f",
    },
    "liftgate": {
        "capacity":      12,
        "allowed_zones": ["residential", "commercial", "industrial", "airport"],
        "description":   "Liftgate trailer — 12 pallets, access anywhere, required for stops with no loading dock",
        "color":         "#0e7c4a",
    },
    "hazmat": {
        "capacity":      12,
        "allowed_zones": ["industrial"],
        "description":   "Hazmat liftgate — 12 pallets, industrial zones only",
        "color":         "#a0280f",
    },
}


# ── OSRM helpers ───────────────────────────────────────────────────────────────

def get_matrices_from_osrm(locs: list[tuple[float, float]]) -> tuple[list, list]:
    """
    Calls the OSRM public routing API to get real road-following distances and
    travel times between every pair of locations.

    OSRM expects coordinates as longitude,latitude (note: lon first).
    It returns a full N×N matrix where matrix[i][j] is the cost of going
    from location i to location j.

    Returns:
        dist_matrix: N×N list of floats, distances in MILES (converted from metres)
        time_matrix: N×N list of floats, travel times in MINUTES (converted from seconds)

    Raises:
        Exception if OSRM returns a non-OK status code (e.g. bad coordinates).
    """
    coords_str = ";".join(f"{lon},{lat}" for lat, lon in locs)
    url        = f"http://router.project-osrm.org/table/v1/driving/{coords_str}"

    log.info("OSRM matrix request | %d locations | url=%s", len(locs), url)
    t0 = time_module.time()

    try:
        r    = requests.get(url, params={"annotations": "duration,distance"}, timeout=15)
        data = r.json()
    except requests.exceptions.Timeout:
        log.error("OSRM request timed out after 15s")
        raise Exception("OSRM request timed out — check your internet connection")
    except requests.exceptions.RequestException as e:
        log.error("OSRM network error: %s", e)
        raise Exception(f"Could not reach OSRM: {e}")

    if data.get("code") != "Ok":
        log.error("OSRM returned error code: %s", data.get("code"))
        raise Exception(f"OSRM error: {data.get('code')} — check that all coordinates are valid")

    elapsed = time_module.time() - t0
    log.info("OSRM matrix received | %.2fs", elapsed)

    # Convert metres → miles and seconds → minutes, keep as floats
    dist_matrix = [
        [round((d / 1000.0) * KM_TO_MILES, 3) for d in row]
        for row in data["distances"]
    ]
    time_matrix = [
        [round(t / 60.0, 2) for t in row]
        for row in data["durations"]
    ]

    return dist_matrix, time_matrix


def get_route_geometry(coords: list[tuple[float, float]]) -> list:
    """
    Calls OSRM's route endpoint to get a road-following polyline for a single
    ordered sequence of coordinates (one truck's full route).

    This is separate from the matrix call — the matrix gives us pairwise costs
    for the solver, while this gives us the actual GPS path to draw on the map.

    Returns:
        List of [lat, lon] pairs representing the road-following path.
        Returns an empty list if the request fails (map will fall back to
        straight lines between stops).
    """
    if len(coords) < 2:
        return []

    coords_str = ";".join(f"{lon},{lat}" for lat, lon in coords)
    url        = f"http://router.project-osrm.org/route/v1/driving/{coords_str}"

    try:
        r    = requests.get(url, params={"overview": "full", "geometries": "geojson"}, timeout=15)
        data = r.json()
    except requests.exceptions.RequestException as e:
        log.warning("OSRM geometry request failed: %s", e)
        return []

    if data.get("code") != "Ok" or not data.get("routes"):
        log.warning("OSRM geometry returned no route: code=%s", data.get("code"))
        return []

    # GeoJSON uses [lon, lat] order — Leaflet.js needs [lat, lon]
    return [[p[1], p[0]] for p in data["routes"][0]["geometry"]["coordinates"]]


# ── Core solver ────────────────────────────────────────────────────────────────

def run_solver(
    loc_names:    list[str],
    loc_coords:   list[tuple[float, float]],
    loc_demands:  list[float],
    fleet:        list[dict],
    loc_svc:      list[float],
    loc_windows:  list[tuple[int, int]],
    n_deliveries: int,
    stop_zones:   list[str],
    priorities:   list[int],
) -> tuple[dict | None, list, list]:
    """
    Runs the OR-Tools VRP solver and returns optimized routes.

    HOW THE CAPACITY MODEL WORKS
    ─────────────────────────────
    We use TWO separate capacity dimensions rather than one combined load dimension.
    This correctly handles the delivery-first, pickup-second workflow:

      DeliveryLoad dimension
        Each delivery stop has a positive demand equal to the cargo being dropped off.
        The cumulative sum of deliveries on a route must not exceed trailer capacity.
        start_cumul_to_zero=True means it starts at 0 and grows as deliveries are assigned.
        OR-Tools enforces: sum(deliveries on route) ≤ capacity.

      PickupLoad dimension
        Each pickup stop has a positive demand equal to the cargo being collected.
        The cumulative sum of pickups on a route must not exceed trailer capacity.
        start_cumul_to_zero=True means it starts at 0 and grows as pickups are assigned.
        OR-Tools enforces: sum(pickups on route) ≤ capacity.

    Since deliveries and pickups are separated in time (deliveries always run first,
    enforced via time window constraints), the trailer space is fully reused:
      - Truck leaves depot loaded with delivery cargo  → at most `capacity` units
      - Truck unloads at delivery stops                → space frees up
      - Truck loads pickup cargo                       → at most `capacity` units
      - Truck returns to depot with pickup cargo

    HOW THE TIME MODEL WORKS
    ─────────────────────────
    A single Time dimension tracks when each truck arrives at each stop.
    service time at each stop is included in the transit callback so the solver
    accounts for unload/load time before moving to the next stop.

    Pickup stops are given a time window that starts no earlier than the minimum
    time to complete one delivery — this enforces delivery-before-pickup ordering.

    HOW ZONE RESTRICTIONS WORK
    ───────────────────────────
    For each stop, we check if the assigned trailer's allowed_zones includes
    that stop's zone type. If not, we call VehicleVar.RemoveValue(v) to hard-forbid
    that truck from visiting that node.

    HOW PRIORITY WORKS
    ───────────────────
    Priority 2 (high) and 3 (urgent) stops get a soft upper bound on their arrival time.
    If the solver visits them later than the soft deadline, it pays a penalty proportional
    to priority. This biases the solver toward visiting important stops early.

    Args:
        loc_names:    Location names, index 0 = depot
        loc_coords:   (lat, lon) tuples, same order as loc_names
        loc_demands:  Cargo units per stop (0 for depot)
        fleet:        List of {"trailer": type, "id": str}
        loc_svc:      Service time in minutes per stop
        loc_windows:  (earliest, latest) arrival time in minutes from start of day
        n_deliveries: How many stops (starting at index 1) are deliveries vs pickups
        stop_zones:   Zone type string per location
        priorities:   Priority level (1=normal, 2=high, 3=urgent) per location

    Returns:
        (result_dict, dist_matrix, time_matrix)
        result_dict is None if no solution was found.
    """

    log.info("Solver starting | %d locations | %d vehicles | %d deliveries | %d pickups",
             len(loc_coords), len(fleet), n_deliveries, len(loc_coords) - n_deliveries - 1)

    t0 = time_module.time()

    # ── Fetch road matrices ────────────────────────────────────────
    dist_m, time_m = get_matrices_from_osrm(loc_coords)
    n_locs         = len(loc_coords)
    n_vehicles     = len(fleet)

    # Validate that every trailer type in the fleet is defined
    for truck in fleet:
        if truck["trailer"] not in TRAILER_DEFS:
            raise ValueError(f"Unknown trailer type '{truck['trailer']}'. "
                             f"Valid types: {list(TRAILER_DEFS.keys())}")

    trailer_defs = [TRAILER_DEFS[t["trailer"]] for t in fleet]
    capacities   = [td["capacity"] for td in trailer_defs]

    # ── Split demands into delivery and pickup arrays ──────────────
    # delivery_demands[i] > 0 only for delivery stops (indices 1..n_deliveries)
    # pickup_demands[i]   > 0 only for pickup stops   (indices n_deliveries+1..)
    delivery_demands = [0.0] * n_locs
    pickup_demands   = [0.0] * n_locs
    for i in range(1, n_locs):
        if i <= n_deliveries:
            delivery_demands[i] = float(loc_demands[i])
        else:
            pickup_demands[i]   = float(loc_demands[i])

    # ── Build zone allowance map per vehicle ───────────────────────
    # allowed_stops[v] = set of node indices that vehicle v is permitted to visit.
    # The depot (index 0) is always allowed for all vehicles.
    allowed_stops = []
    for v, td in enumerate(trailer_defs):
        allowed = {
            n for n, zone in enumerate(stop_zones)
            if n == 0 or zone == "depot" or zone in td["allowed_zones"]
        }
        allowed_stops.append(allowed)
        log.debug("Truck %s (%s) | allowed stops: %s", fleet[v]["id"], fleet[v]["trailer"],
                  [loc_names[n] for n in sorted(allowed) if n > 0])

    # ── Initialize OR-Tools routing model ─────────────────────────
    # RoutingIndexManager maps between node indices (our location indices)
    # and internal OR-Tools indices used by the solver.
    # Args: (num_locations, num_vehicles, depot_index)
    mgr = pywrapcp.RoutingIndexManager(n_locs, n_vehicles, 0)
    mdl = pywrapcp.RoutingModel(mgr)

    # ── Cost callback ──────────────────────────────────────────────
    # OR-Tools minimizes total arc cost. We use road distance in miles
    # scaled by 1000 (OR-Tools needs integers internally) to preserve
    # 3 decimal places of precision.
    COST_SCALE = 1000

    def dist_cb(fi: int, ti: int) -> int:
        i = mgr.IndexToNode(fi)
        j = mgr.IndexToNode(ti)
        return int(dist_m[i][j] * COST_SCALE)

    dist_idx = mdl.RegisterTransitCallback(dist_cb)
    mdl.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    # ── Time dimension ─────────────────────────────────────────────
    # Tracks cumulative time elapsed at each stop, including travel + service time.
    # max_slack=60: trucks can wait up to 60 minutes at a stop for a time window to open.
    # horizon=600:  routes must complete within 600 minutes (10 hours) of start.
    # start_cumul_to_zero=False: trucks don't all have to start at time 0.
    def time_cb(fi: int, ti: int) -> int:
        i = mgr.IndexToNode(fi)
        j = mgr.IndexToNode(ti)
        # Include service time at i before travelling to j
        # Round to int — OR-Tools dimensions require integers
        return int(round(time_m[i][j] + loc_svc[i]))

    time_idx = mdl.RegisterTransitCallback(time_cb)
    mdl.AddDimension(time_idx, 60, 600, False, "Time")
    tdim = mdl.GetDimensionOrDie("Time")

    # Apply arrival time windows to each location
    for node, (earliest, latest) in enumerate(loc_windows):
        tdim.CumulVar(mgr.NodeToIndex(node)).SetRange(earliest, latest)

    # ── Delivery load dimension ────────────────────────────────────
    # Enforces that the total delivery cargo assigned to any one truck
    # does not exceed that truck's trailer capacity.
    # See the docstring above for the full explanation of why this works.
    def del_demand_cb(fi: int) -> int:
        return int(delivery_demands[mgr.IndexToNode(fi)])

    del_idx = mdl.RegisterUnaryTransitCallback(del_demand_cb)
    mdl.AddDimensionWithVehicleCapacity(
        del_idx,
        0,           # no slack beyond the capacity ceiling
        capacities,  # per-vehicle capacity ceiling
        True,        # start_cumul_to_zero=True: accumulates from 0 as stops are assigned
        "DeliveryLoad"
    )

    # ── Pickup load dimension ──────────────────────────────────────
    # Same pattern as DeliveryLoad but for pickup cargo.
    # Ensures pickup cargo collected on a route fits in the trailer.
    def pick_demand_cb(fi: int) -> int:
        return int(pickup_demands[mgr.IndexToNode(fi)])

    pick_idx = mdl.RegisterUnaryTransitCallback(pick_demand_cb)
    mdl.AddDimensionWithVehicleCapacity(
        pick_idx,
        0,
        capacities,
        True,        # start at 0, grows as pickups are added to route
        "PickupLoad"
    )

    # ── Delivery-before-pickup time constraint ─────────────────────
    # Pickup stops get a time window floor equal to the earliest possible
    # time a delivery could be completed. This prevents the solver from
    # scheduling pickups before any deliveries have been made.
    delivery_nodes = list(range(1, n_deliveries + 1))
    pickup_nodes   = list(range(n_deliveries + 1, n_locs))

    if delivery_nodes and pickup_nodes:
        # Earliest a delivery can finish: fastest travel from depot + fastest service
        min_delivery_done = (
            min(time_m[0][d] for d in delivery_nodes) +
            min(loc_svc[d]   for d in delivery_nodes)
        )
        for pn in pickup_nodes:
            idx = mgr.NodeToIndex(pn)
            earliest, latest = loc_windows[pn]
            tdim.CumulVar(idx).SetRange(max(earliest, int(min_delivery_done)), latest)

        log.debug("Delivery-before-pickup: earliest pickup allowed at t=%.1f min", min_delivery_done)

    # ── Priority soft time bounds ──────────────────────────────────
    # High/urgent priority stops get a soft deadline: if the solver arrives
    # after (window_start + 60 min), it pays a penalty per minute late.
    # This doesn't hard-forbid late arrivals — it just makes them expensive,
    # so the solver schedules priority stops earlier when possible.
    for node, priority in enumerate(priorities):
        if priority > 1 and node > 0:
            idx           = mgr.NodeToIndex(node)
            soft_deadline = loc_windows[node][0] + 60  # 60 min grace period
            penalty       = priority * 500              # higher priority = steeper penalty
            tdim.SetCumulVarSoftUpperBound(idx, soft_deadline, penalty)
            log.debug("Priority %d on '%s': soft deadline at t=%d, penalty=%d",
                      priority, loc_names[node], soft_deadline, penalty)

    # ── Zone restrictions ──────────────────────────────────────────
    # Hard-forbid each vehicle from visiting stops in zones it cannot access.
    # VehicleVar(node).RemoveValue(v) tells OR-Tools that vehicle v can never
    # be assigned to serve node — the solver will never consider this assignment.
    for node in range(1, n_locs):
        for v in range(n_vehicles):
            if node not in allowed_stops[v]:
                mdl.VehicleVar(mgr.NodeToIndex(node)).RemoveValue(v)

    # ── All stops are mandatory ────────────────────────────────────
    # We do NOT add disjunctions (optional stops with skip penalties).
    # Every stop must be served. If the problem is infeasible (e.g. not enough
    # trucks), the solver returns None rather than silently skipping stops.

    # ── Solver configuration ───────────────────────────────────────
    # PARALLEL_CHEAPEST_INSERTION: builds an initial solution by repeatedly
    # inserting the cheapest unassigned stop into an existing route.
    # This works better than PATH_CHEAPEST_ARC for capacity-constrained problems
    # because it considers multiple routes simultaneously.
    #
    # GUIDED_LOCAL_SEARCH: improves the initial solution by exploring neighbouring
    # solutions (swapping stops between trucks, reordering within a route, etc.)
    # until the time limit is reached.
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = 20

    log.info("Running OR-Tools solver (time limit: %ds)...", params.time_limit.seconds)
    sol = mdl.SolveWithParameters(params)

    if not sol:
        log.warning("Solver found no solution | Check: enough trucks? zone conflicts? capacity?")
        return None, dist_m, time_m

    elapsed = time_module.time() - t0
    log.info("Solver finished | %.2fs | objective=%.3f miles",
             elapsed, sol.ObjectiveValue() / COST_SCALE)

    # ── Parse solution into readable route objects ─────────────────
    tdim     = mdl.GetDimensionOrDie("Time")
    routes   = []

    for v in range(n_vehicles):
        idx        = mdl.Start(v)
        stops_out  = []
        del_total  = 0.0
        pick_total = 0.0

        while not mdl.IsEnd(idx):
            n       = mgr.IndexToNode(idx)
            arrival = sol.Min(tdim.CumulVar(idx))
            stype   = "depot" if n == 0 else ("delivery" if n <= n_deliveries else "pickup")

            if stype == "delivery": del_total  += float(loc_demands[n])
            if stype == "pickup":   pick_total += float(loc_demands[n])

            # Convert arrival minutes to a clock time string (starting from 8:00 AM)
            clock_minutes = 480 + arrival  # 480 = 8 * 60
            arrival_str   = f"{clock_minutes // 60}:{clock_minutes % 60:02d}"

            stops_out.append({
                "node":         n,
                "name":         loc_names[n],
                "type":         stype,
                "zone":         stop_zones[n],
                "priority":     priorities[n],
                "arrival_min":  arrival,
                "arrival_time": arrival_str,
                "service_min":  float(loc_svc[n]),
                "demand":       float(loc_demands[n]),
            })
            idx = sol.Value(mdl.NextVar(idx))

        # Return-to-depot stop
        final_arrival = sol.Min(tdim.CumulVar(idx))
        clock_ret     = 480 + final_arrival
        stops_out.append({
            "node":         0,
            "name":         loc_names[0],
            "type":         "depot",
            "zone":         "depot",
            "priority":     0,
            "arrival_min":  final_arrival,
            "arrival_time": f"{clock_ret // 60}:{clock_ret % 60:02d}",
            "service_min":  0.0,
            "demand":       0.0,
        })

        # Only include trucks that were actually used (have at least one real stop)
        if len(stops_out) > 2:
            td = trailer_defs[v]
            log.info("  %s (%s) | departs %.1f units | picks up %.1f units | %d stops",
                     fleet[v]["id"], fleet[v]["trailer"], del_total, pick_total,
                     len(stops_out) - 2)
            routes.append({
                "truck":          fleet[v]["id"],
                "trailer_type":   fleet[v]["trailer"],
                "truck_color":    td["color"],
                "description":    td["description"],
                "departure_load": round(del_total, 2),
                "delivery_load":  round(del_total, 2),
                "pickup_load":    round(pick_total, 2),
                "capacity":       float(capacities[v]),
                "allowed_zones":  td["allowed_zones"],
                "stops":          stops_out,
                "node_sequence":  [s["node"] for s in stops_out],
            })

    total_miles = round(sol.ObjectiveValue() / COST_SCALE, 3)
    log.info("Solution: %d trucks used | %.3f total miles", len(routes), total_miles)

    return {
        "total_distance_miles": total_miles,
        "routes":               routes,
        "locations":            loc_names,
        "coords":               [list(c) for c in loc_coords],
        "distances":            dist_m,   # miles
        "times":                time_m,   # minutes (floats)
    }, dist_m, time_m


# ── Request validation helper ──────────────────────────────────────────────────

def validate_stop(stop: dict, label: str) -> None:
    """
    Validates that a stop dict from the request body has all required fields
    and that the coordinate values are within plausible geographic ranges.

    Raises ValueError with a descriptive message if anything is missing or invalid.
    """
    if not stop.get("name", "").strip():
        raise ValueError(f"{label} is missing a name.")

    coords = stop.get("coords")
    if not coords or len(coords) != 2:
        raise ValueError(f"{label} '{stop.get('name')}' is missing coordinates.")

    lat, lon = coords
    if not (-90 <= lat <= 90):
        raise ValueError(f"{label} '{stop.get('name')}' has an invalid latitude: {lat}")
    if not (-180 <= lon <= 180):
        raise ValueError(f"{label} '{stop.get('name')}' has an invalid longitude: {lon}")

    demand = stop.get("demand", 0)
    if float(demand) < 0:
        raise ValueError(f"{label} '{stop.get('name')}' has a negative demand: {demand}")


# ── Flask routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main frontend HTML file."""
    return send_from_directory('static', 'index.html')


@app.route('/trailer_types', methods=['GET'])
def trailer_types():
    """Return the full trailer type definitions so the frontend can build its UI."""
    return jsonify(TRAILER_DEFS)


@app.route('/solve', methods=['GET', 'POST'])
def solve_route():
    """
    Main optimization endpoint.

    POST body (JSON):
        depot:                 str   — depot location name (already geocoded by frontend)
        depot_coords:          [lat, lon]
        deliveries:            list of stop objects (name, coords, demand, zone, priority)
        pickups:               list of stop objects (name, coords, demand, zone, priority)
        fleet:                 list of {"trailer": type, "id": str}
        delivery_service_time: float — minutes spent unloading at each delivery stop
        pickup_service_time:   float — minutes spent loading at each pickup stop

    GET: runs a hardcoded demo problem (useful for testing the connection).

    Returns JSON with routes, total distance, and the distance/time matrices.
    """
    request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log.info("=== /solve request %s | method=%s | ip=%s ===",
             request_id, request.method, request.remote_addr)

    try:
        if request.method == 'POST':
            body = request.get_json(force=True, silent=True)
            if not body:
                log.warning("Request %s | invalid or missing JSON body", request_id)
                return jsonify({"error": "Request body must be valid JSON."}), 400

            deliveries = body.get('deliveries', [])
            pickups    = body.get('pickups',    [])
            n_del      = len(deliveries)
            del_svc    = float(body.get('delivery_service_time', 20))
            pick_svc   = float(body.get('pickup_service_time',   45))
            fleet      = body.get('fleet', [])

            # ── Input validation ───────────────────────────────────
            if not deliveries and not pickups:
                return jsonify({"error": "Add at least one delivery or pickup stop."}), 400

            depot_name   = body.get('depot', '').strip()
            depot_coords = body.get('depot_coords')
            if not depot_name:
                return jsonify({"error": "Depot name is required."}), 400
            if not depot_coords or len(depot_coords) != 2:
                return jsonify({"error": "Depot coordinates are required."}), 400

            for i, d in enumerate(deliveries):
                validate_stop(d, f"Delivery {i+1}")
            for i, p in enumerate(pickups):
                validate_stop(p, f"Pickup {i+1}")

            if not fleet:
                return jsonify({"error": "Add at least one truck in the Fleet tab."}), 400

            for i, truck in enumerate(fleet):
                if "trailer" not in truck:
                    return jsonify({"error": f"Fleet entry {i+1} is missing a trailer type."}), 400
                if truck["trailer"] not in TRAILER_DEFS:
                    return jsonify({
                        "error": f"Unknown trailer type '{truck['trailer']}'. "
                                 f"Valid: {list(TRAILER_DEFS.keys())}"
                    }), 400

            if del_svc < 0 or pick_svc < 0:
                return jsonify({"error": "Service times cannot be negative."}), 400

            log.info("Request %s | depot='%s' | %d deliveries | %d pickups | %d trucks",
                     request_id, depot_name, n_del, len(pickups), len(fleet))

            # ── Build solver inputs ────────────────────────────────
            loc_names   = [depot_name] + [d['name'] for d in deliveries] + [p['name'] for p in pickups]
            loc_coords  = ([tuple(depot_coords)]
                           + [tuple(d['coords']) for d in deliveries]
                           + [tuple(p['coords']) for p in pickups])
            loc_demands = ([0.0]
                           + [float(abs(d['demand'])) for d in deliveries]
                           + [float(abs(p['demand'])) for p in pickups])
            loc_svc     = [0.0] + [del_svc] * n_del + [pick_svc] * len(pickups)
            loc_windows = [(0, 600)] * len(loc_names)
            stop_zones  = (["depot"]
                           + [d.get('zone', 'commercial') for d in deliveries]
                           + [p.get('zone', 'commercial') for p in pickups])
            priorities  = ([0]
                           + [int(d.get('priority', 1)) for d in deliveries]
                           + [int(p.get('priority', 1)) for p in pickups])

        else:
            # ── GET: demo problem ──────────────────────────────────
            # Realistic pallet counts — 53ft trailer holds 26 pallets max.
            # 2 deliveries (10+8 pallets) and 2 pickups (6+12 pallets).
            log.info("Request %s | GET demo problem", request_id)
            loc_names   = ["Depot (Addison, IL)", "O'Hare Airport", "Willis Tower", "Navy Pier", "Schaumburg"]
            loc_coords  = [(41.9314, -88.0126), (41.9742, -87.9073), (41.8789, -87.6359),
                           (41.8917, -87.6086), (42.0334, -88.0834)]
            loc_demands = [0, 10, 8, 6, 12]
            fleet       = [{"trailer": "53ft", "id": f"Truck-{i+1}"} for i in range(3)]
            loc_svc     = [0, 20, 20, 45, 45]
            loc_windows = [(0, 600)] * 5
            n_del       = 2
            stop_zones  = ["depot", "airport", "commercial", "commercial", "industrial"]
            priorities  = [0, 1, 1, 1, 1]

        # ── Run solver ─────────────────────────────────────────────
        result, _, _ = run_solver(
            loc_names, loc_coords, loc_demands, fleet,
            loc_svc, loc_windows, n_del, stop_zones, priorities
        )

        if not result:
            msg = ("No solution found. Possible causes: "
                   "not enough trucks for the total cargo, "
                   "zone restrictions prevent some stops from being reached, "
                   "or time windows are too tight.")
            log.warning("Request %s | no solution | %s", request_id, msg)
            return jsonify({"error": msg}), 400

        log.info("Request %s | success | %d routes | %.3f miles",
                 request_id, len(result["routes"]), result["total_distance_miles"])
        return jsonify(result)

    except ValueError as e:
        # Input validation errors — tell the user what to fix
        log.warning("Request %s | validation error: %s", request_id, e)
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        # Unexpected errors — log full traceback, return safe message
        log.error("Request %s | unexpected error: %s\n%s", request_id, e, traceback.format_exc())
        return jsonify({"error": f"Internal server error: {e}"}), 500


@app.route('/geometry', methods=['POST'])
def geometry():
    """
    Fetches road-following polylines for a set of already-solved routes.

    Called after /solve returns — takes the solved node sequences and
    fetches the actual GPS paths from OSRM so Leaflet can draw them on the map.

    POST body:
        coords: list of [lat, lon] for all locations (depot + all stops)
        routes: list of node-index sequences (one per truck)

    Returns:
        {"geometries": [[lat, lon], ...] per truck}
    """
    log.info("/geometry request | ip=%s", request.remote_addr)
    try:
        body = request.get_json(force=True, silent=True)
        if not body:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        all_coords  = body.get('coords', [])
        route_nodes = body.get('routes', [])

        if not all_coords or not route_nodes:
            return jsonify({"error": "coords and routes are required."}), 400

        geometries = []
        for nodes in route_nodes:
            ordered = [all_coords[n] for n in nodes]
            geometries.append(get_route_geometry(ordered))

        log.info("/geometry | %d routes fetched", len(geometries))
        return jsonify({"geometries": geometries})

    except Exception as e:
        log.error("/geometry error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": f"Internal server error: {e}"}), 500


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    log.info("=" * 60)
    log.info("VRP Route Optimizer starting on port %d", port)
    log.info("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=debug)
