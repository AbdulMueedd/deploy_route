"""
VRP Route Optimizer — Flask Backend
The Cary Company

Solves a Vehicle Routing Problem (VRP) with:
  - Pickup and delivery stops (ALL deliveries on a truck finish before ANY pickup on that truck)
  - Per-stop time windows (e.g. "Company A only accepts deliveries 12pm-1pm")
  - Trailer-based capacity constraints (each trailer type has its own capacity and zone access)
  - Priority stops (high/urgent stops are penalized if visited late)
  - Zone restrictions (trailer type determines which zones it can access)
  - Real road distances and travel times via OSRM
  - OR-Tools for route optimization

Units: distances in miles, time in minutes, cargo measured in pallets.

TIME MODEL
──────────
All times are minutes from the start of the working day (t=0 means 8:00 AM).
So a window of "12pm-1pm" becomes (240, 300):
    12:00 PM = 4 hours after 8:00 AM = 240 minutes
    1:00 PM  = 5 hours after 8:00 AM = 300 minutes
The frontend converts clock times to these minute offsets before sending.
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

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/vrp_audit.log', encoding='utf-8'),
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

DAY_START_MIN = 0     # t=0 corresponds to 8:00 AM (see DAY_START_CLOCK)
DAY_END_MIN   = 600   # 600 minutes = 10 hour working day, ends 6:00 PM
DAY_START_CLOCK = 480 # 8:00 AM in minutes-from-midnight, used for display only

TRAILER_DEFS = {
    "53ft": {
        "capacity":      26,
        "allowed_zones": ["residential", "commercial", "industrial", "airport"],
        "description":   "53' dry van — 26 pallets, no zone restrictions",
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

def get_matrices_from_osrm(locs):
    """
    Calls the OSRM public routing API to get real road-following distances and
    travel times between every pair of locations.

    Returns:
        dist_matrix: N×N list, distances in MILES (rounded to 1 decimal)
        time_matrix: N×N list, travel times in MINUTES (rounded to nearest minute)
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

    dist_matrix = [
        [round((d / 1000) * KM_TO_MILES, 1) for d in row]
        for row in data["distances"]
    ]
    time_matrix = [
        [round(t / 60) for t in row]
        for row in data["durations"]
    ]

    return dist_matrix, time_matrix


def get_route_geometry(coords):
    """Calls OSRM's route endpoint to get a road-following polyline for one truck's route."""
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

    return [[p[1], p[0]] for p in data["routes"][0]["geometry"]["coordinates"]]


# ── Pre-solve feasibility checks ────────────────────────────────────────────────
# These run BEFORE the solver so an impossible problem returns an instant, specific
# error message rather than a 20-second "no solution found" timeout.

def check_feasibility(loc_names, loc_demands, loc_windows, loc_svc, stop_zones,
                      fleet, time_m, n_deliveries):
    """
    Runs a series of fast feasibility checks before the solver starts.
    Returns a list of human-readable error strings (empty if everything is feasible).
    """
    errors = []
    capacities = [TRAILER_DEFS[t["trailer"]]["capacity"] for t in fleet]
    max_cap    = max(capacities) if capacities else 0
    total_cap  = sum(capacities)

    # 1. Total cargo vs fleet capacity (deliveries and pickups checked separately,
    #    because the two-dimension model enforces them independently)
    total_delivery = sum(loc_demands[i] for i in range(1, n_deliveries + 1))
    total_pickup   = sum(loc_demands[i] for i in range(n_deliveries + 1, len(loc_demands)))
    if total_delivery > total_cap:
        errors.append(f"Total delivery cargo ({total_delivery} pallets) exceeds combined "
                      f"fleet capacity ({total_cap} pallets). Add more trucks or reduce load.")
    if total_pickup > total_cap:
        errors.append(f"Total pickup cargo ({total_pickup} pallets) exceeds combined "
                      f"fleet capacity ({total_cap} pallets). Add more trucks or reduce load.")

    # 2. Any single stop heavier than the largest trailer
    for i in range(1, len(loc_demands)):
        if loc_demands[i] > max_cap:
            errors.append(f"Stop '{loc_names[i]}' needs {int(loc_demands[i])} pallets, but the "
                          f"largest trailer only holds {max_cap}. No single truck can serve it.")

    # 3. Zone reachability — every stop needs at least one truck whose trailer allows its zone
    for i in range(1, len(stop_zones)):
        zone = stop_zones[i]
        eligible = [t for t in fleet if zone in TRAILER_DEFS[t["trailer"]]["allowed_zones"]]
        if not eligible:
            errors.append(f"Stop '{loc_names[i]}' is in zone '{zone}', but no truck in the fleet "
                          f"has a trailer that can access it.")

    # 4. Time window reachability — can the truck physically reach the stop before its window closes?
    for i in range(1, len(loc_windows)):
        earliest, latest = loc_windows[i]
        # Fastest possible arrival = straight drive from depot (ignores other stops, service time)
        min_arrival = time_m[0][i]
        if min_arrival > latest:
            errors.append(f"Stop '{loc_names[i]}' closes at {fmt_clock(latest)} but the fastest "
                          f"possible arrival from the depot is {fmt_clock(int(min_arrival))}. "
                          f"Window is unreachable.")
        if earliest > DAY_END_MIN:
            errors.append(f"Stop '{loc_names[i]}' opens at {fmt_clock(earliest)}, which is after "
                          f"the end of the working day ({fmt_clock(DAY_END_MIN)}).")
        if earliest > latest:
            errors.append(f"Stop '{loc_names[i]}' has an invalid window: opens at "
                          f"{fmt_clock(earliest)} but closes at {fmt_clock(latest)}.")

    return errors


def fmt_clock(minutes_from_start):
    """Convert minutes-from-day-start into a readable clock time like '12:30 PM'."""
    total = DAY_START_CLOCK + minutes_from_start
    h24   = (total // 60) % 24
    m     = total % 60
    suffix = "AM" if h24 < 12 else "PM"
    h12 = h24 % 12
    if h12 == 0:
        h12 = 12
    return f"{h12}:{m:02d} {suffix}"


# ── Core solver ────────────────────────────────────────────────────────────────

def run_solver(loc_names, loc_coords, loc_demands, fleet, loc_svc, loc_windows,
               n_deliveries, stop_zones, priorities):
    """
    Runs the OR-Tools VRP solver and returns optimized routes.

    CAPACITY MODEL — two independent dimensions
    ───────────────────────────────────────────
    DeliveryLoad: sum of delivery cargo on a route ≤ trailer capacity.
    PickupLoad:   sum of pickup cargo on a route   ≤ trailer capacity.
    Tracked separately so the two can never be traded against each other.

    TIME MODEL — per-stop windows
    ──────────────────────────────
    Each stop has its own [earliest, latest] arrival window in minutes from day start.
    The solver hard-enforces these — a truck must arrive within the window, waiting
    if it gets there early (up to the max_slack), and the assignment is infeasible
    if it cannot arrive before the window closes.

    HARD DELIVERY-BEFORE-PICKUP ORDERING
    ─────────────────────────────────────
    For every (delivery d, pickup p) pair, we add a conditional constraint:
        IF the same truck serves both d and p
        THEN arrival_time[p] ≥ arrival_time[d] + service_time[d]
    Applied across all pairs, this forces a truck to complete EVERY delivery on its
    route before starting ANY pickup — even if it has spare capacity. A truck cannot
    interleave pickups between deliveries.

    ZONE RESTRICTIONS — VehicleVar.RemoveValue() hard-forbids illegal truck/stop pairs.
    PRIORITY — soft upper bound on arrival time, penalty proportional to priority.
    """
    log.info("Solver starting | %d locations | %d vehicles | %d deliveries | %d pickups",
             len(loc_coords), len(fleet), n_deliveries, len(loc_coords) - n_deliveries - 1)

    t0 = time_module.time()

    dist_m, time_m = get_matrices_from_osrm(loc_coords)
    n_locs         = len(loc_coords)
    n_vehicles     = len(fleet)

    for truck in fleet:
        if truck["trailer"] not in TRAILER_DEFS:
            raise ValueError(f"Unknown trailer type '{truck['trailer']}'. "
                             f"Valid types: {list(TRAILER_DEFS.keys())}")

    trailer_defs = [TRAILER_DEFS[t["trailer"]] for t in fleet]
    capacities   = [td["capacity"] for td in trailer_defs]

    # ── Split demands into delivery and pickup arrays ──────────────
    delivery_demands = [0] * n_locs
    pickup_demands   = [0] * n_locs
    for i in range(1, n_locs):
        if i <= n_deliveries:
            delivery_demands[i] = int(loc_demands[i])
        else:
            pickup_demands[i]   = int(loc_demands[i])

    # ── Zone allowance map per vehicle ─────────────────────────────
    allowed_stops = []
    for v, td in enumerate(trailer_defs):
        allowed = {
            n for n, zone in enumerate(stop_zones)
            if n == 0 or zone == "depot" or zone in td["allowed_zones"]
        }
        allowed_stops.append(allowed)

    # ── Routing model ──────────────────────────────────────────────
    mgr = pywrapcp.RoutingIndexManager(n_locs, n_vehicles, 0)
    mdl = pywrapcp.RoutingModel(mgr)

    # ── Cost callback (distance in miles, scaled to integers) ──────
    COST_SCALE = 1000

    def dist_cb(fi, ti):
        i = mgr.IndexToNode(fi)
        j = mgr.IndexToNode(ti)
        return int(dist_m[i][j] * COST_SCALE)

    dist_idx = mdl.RegisterTransitCallback(dist_cb)
    mdl.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    # ── Time dimension ─────────────────────────────────────────────
    # max_slack=600 allows a truck to wait for a window to open (e.g. arrive at
    # 11:00 for a 12:00 window and wait an hour). horizon=DAY_END_MIN caps the day.
    def time_cb(fi, ti):
        i = mgr.IndexToNode(fi)
        j = mgr.IndexToNode(ti)
        return int(time_m[i][j] + loc_svc[i])

    time_idx = mdl.RegisterTransitCallback(time_cb)
    mdl.AddDimension(
        time_idx,
        600,           # max_slack — how long a truck may wait for a window to open
        DAY_END_MIN,   # horizon — route must finish within the working day
        False,         # don't force start cumul to zero (trucks may start later)
        "Time"
    )
    tdim = mdl.GetDimensionOrDie("Time")

    # ── Per-stop time windows ──────────────────────────────────────
    # Each stop is constrained to its own [earliest, latest] arrival window.
    for node, (earliest, latest) in enumerate(loc_windows):
        tdim.CumulVar(mgr.NodeToIndex(node)).SetRange(int(earliest), int(latest))

    # ── Capacity dimensions ────────────────────────────────────────
    def del_demand_cb(fi):
        return int(delivery_demands[mgr.IndexToNode(fi)])

    del_idx = mdl.RegisterUnaryTransitCallback(del_demand_cb)
    mdl.AddDimensionWithVehicleCapacity(del_idx, 0, capacities, True, "DeliveryLoad")

    def pick_demand_cb(fi):
        return int(pickup_demands[mgr.IndexToNode(fi)])

    pick_idx = mdl.RegisterUnaryTransitCallback(pick_demand_cb)
    mdl.AddDimensionWithVehicleCapacity(pick_idx, 0, capacities, True, "PickupLoad")

    # ── HARD delivery-before-pickup ordering ───────────────────────
    # Implemented with a "Stage" counter dimension.
    #
    # The Stage dimension increments by 1 every time a truck visits a pickup stop,
    # so its cumulative value at any node = number of pickups that truck has already
    # visited. We then force every delivery to be visited while Stage == 0 — meaning
    # NO pickup can have happened before it.
    #
    # Because each vehicle accumulates its own Stage count, this is enforced
    # per-truck automatically: every truck must complete ALL of its deliveries
    # (Stage still 0) before it touches ANY pickup (which pushes Stage to 1+).
    # A truck cannot interleave a pickup between deliveries even with spare capacity.
    #
    # This counter approach is used instead of time-based precedence constraints
    # because it propagates cleanly through OR-Tools' search without conflicting
    # with the Time dimension.
    def stage_cb(fi):
        n = mgr.IndexToNode(fi)
        return 1 if n > n_deliveries else 0   # pickups add 1, deliveries/depot add 0

    stage_idx = mdl.RegisterUnaryTransitCallback(stage_cb)
    mdl.AddDimension(stage_idx, 0, n_locs, True, "Stage")
    stage = mdl.GetDimensionOrDie("Stage")

    # Every delivery must be served before any pickup → Stage cumul must be 0 there
    for d in range(1, n_deliveries + 1):
        stage.CumulVar(mgr.NodeToIndex(d)).SetValue(0)

    # ── Priority soft time bounds ──────────────────────────────────
    for node, priority in enumerate(priorities):
        if priority > 1 and node > 0:
            idx           = mgr.NodeToIndex(node)
            soft_deadline = loc_windows[node][0] + 60
            penalty       = priority * 500
            tdim.SetCumulVarSoftUpperBound(idx, int(soft_deadline), penalty)

    # ── Zone restrictions ──────────────────────────────────────────
    for node in range(1, n_locs):
        for v in range(n_vehicles):
            if node not in allowed_stops[v]:
                mdl.VehicleVar(mgr.NodeToIndex(node)).RemoveValue(v)

    # ── Solver configuration ───────────────────────────────────────
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
        log.warning("Solver found no solution | Check: time windows, capacity, zones, ordering")
        return None, dist_m, time_m

    elapsed = time_module.time() - t0
    log.info("Solver finished | %.2fs | objective=%.3f miles",
             elapsed, sol.ObjectiveValue() / COST_SCALE)

    # ── Parse solution ─────────────────────────────────────────────
    tdim   = mdl.GetDimensionOrDie("Time")
    routes = []

    for v in range(n_vehicles):
        idx        = mdl.Start(v)
        stops_out  = []
        del_total  = 0
        pick_total = 0

        while not mdl.IsEnd(idx):
            n       = mgr.IndexToNode(idx)
            arrival = sol.Min(tdim.CumulVar(idx))
            stype   = "depot" if n == 0 else ("delivery" if n <= n_deliveries else "pickup")

            if stype == "delivery": del_total  += int(loc_demands[n])
            if stype == "pickup":   pick_total += int(loc_demands[n])

            clock_minutes = DAY_START_CLOCK + arrival
            arrival_str   = f"{clock_minutes // 60}:{clock_minutes % 60:02d}"
            win_e, win_l  = loc_windows[n]

            stops_out.append({
                "node":          n,
                "name":          loc_names[n],
                "type":          stype,
                "zone":          stop_zones[n],
                "priority":      priorities[n],
                "arrival_min":   arrival,
                "arrival_time":  arrival_str,
                "window_open":   int(win_e),
                "window_close":  int(win_l),
                "window_label":  f"{fmt_clock(int(win_e))} – {fmt_clock(int(win_l))}",
                "service_min":   int(loc_svc[n]),
                "demand":        int(loc_demands[n]),
            })
            idx = sol.Value(mdl.NextVar(idx))

        final_arrival = sol.Min(tdim.CumulVar(idx))
        clock_ret     = DAY_START_CLOCK + final_arrival
        stops_out.append({
            "node":         0,
            "name":         loc_names[0],
            "type":         "depot",
            "zone":         "depot",
            "priority":     0,
            "arrival_min":  final_arrival,
            "arrival_time": f"{clock_ret // 60}:{clock_ret % 60:02d}",
            "window_open":  0,
            "window_close": DAY_END_MIN,
            "window_label": "",
            "service_min":  0,
            "demand":       0,
        })

        if len(stops_out) > 2:
            td = trailer_defs[v]
            log.info("  %s (%s) | departs %d pallets | picks up %d pallets | %d stops",
                     fleet[v]["id"], fleet[v]["trailer"], del_total, pick_total,
                     len(stops_out) - 2)
            routes.append({
                "truck":          fleet[v]["id"],
                "trailer_type":   fleet[v]["trailer"],
                "truck_color":    td["color"],
                "description":    td["description"],
                "departure_load": del_total,
                "delivery_load":  del_total,
                "pickup_load":    pick_total,
                "capacity":       capacities[v],
                "allowed_zones":  td["allowed_zones"],
                "stops":          stops_out,
                "node_sequence":  [s["node"] for s in stops_out],
            })

    total_miles = round(sol.ObjectiveValue() / COST_SCALE, 1)
    log.info("Solution: %d trucks used | %.1f total miles", len(routes), total_miles)

    return {
        "total_distance_miles": total_miles,
        "routes":               routes,
        "locations":            loc_names,
        "coords":               [list(c) for c in loc_coords],
        "distances":            dist_m,
        "times":                time_m,
    }, dist_m, time_m


# ── Request validation helper ──────────────────────────────────────────────────

def validate_stop(stop, label):
    """Validates a single stop dict from the request body."""
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
    if int(demand) < 0:
        raise ValueError(f"{label} '{stop.get('name')}' has a negative demand: {demand}")

    # Validate time window if provided
    we = stop.get("window_open")
    wl = stop.get("window_close")
    if we is not None and wl is not None:
        if int(we) > int(wl):
            raise ValueError(f"{label} '{stop.get('name')}' has an invalid time window: "
                             f"opens after it closes.")


def parse_window(stop):
    """
    Extract a stop's time window in minutes-from-day-start.
    Defaults to the full working day (0, DAY_END_MIN) if not specified.
    The frontend sends window_open / window_close as minute offsets.
    """
    we = stop.get("window_open")
    wl = stop.get("window_close")
    earliest = int(we) if we is not None else DAY_START_MIN
    latest   = int(wl) if wl is not None else DAY_END_MIN
    return (earliest, latest)


# ── Flask routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/trailer_types', methods=['GET'])
def trailer_types():
    return jsonify(TRAILER_DEFS)


@app.route('/solve', methods=['GET', 'POST'])
def solve_route():
    """
    Main optimization endpoint.

    POST body (JSON):
        depot:                 str
        depot_coords:          [lat, lon]
        deliveries:            list of {name, coords, demand, zone, priority,
                                        window_open?, window_close?}
        pickups:               same shape as deliveries
        fleet:                 list of {"trailer": type, "id": str}
        delivery_service_time: int (minutes)
        pickup_service_time:   int (minutes)

    window_open / window_close are minutes from 8:00 AM (t=0).
    Example: a 12pm-1pm window is window_open=240, window_close=300.
    They are optional — omitted stops default to the full working day.
    """
    request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log.info("=== /solve request %s | method=%s | ip=%s ===",
             request_id, request.method, request.remote_addr)

    try:
        if request.method == 'POST':
            body = request.get_json(force=True, silent=True)
            if not body:
                return jsonify({"error": "Request body must be valid JSON."}), 400

            deliveries = body.get('deliveries', [])
            pickups    = body.get('pickups',    [])
            n_del      = len(deliveries)
            del_svc    = int(body.get('delivery_service_time', 20))
            pick_svc   = int(body.get('pickup_service_time',   45))
            fleet      = body.get('fleet', [])

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
            loc_demands = ([0]
                           + [int(abs(d['demand'])) for d in deliveries]
                           + [int(abs(p['demand'])) for p in pickups])
            loc_svc     = [0] + [del_svc] * n_del + [pick_svc] * len(pickups)
            # Per-stop time windows (depot is always open the full day)
            loc_windows = ([(DAY_START_MIN, DAY_END_MIN)]
                           + [parse_window(d) for d in deliveries]
                           + [parse_window(p) for p in pickups])
            stop_zones  = (["depot"]
                           + [d.get('zone', 'commercial') for d in deliveries]
                           + [p.get('zone', 'commercial') for p in pickups])
            priorities  = ([0]
                           + [int(d.get('priority', 1)) for d in deliveries]
                           + [int(p.get('priority', 1)) for p in pickups])

            # ── Pre-solve feasibility check (needs the time matrix) ──
            # Fetch the matrix once here so we can validate windows before solving.
            dist_m, time_m = get_matrices_from_osrm(loc_coords)
            feas_errors = check_feasibility(loc_names, loc_demands, loc_windows, loc_svc,
                                            stop_zones, fleet, time_m, n_del)
            if feas_errors:
                log.warning("Request %s | infeasible before solve | %s", request_id, feas_errors)
                return jsonify({
                    "error": "The problem is infeasible:\n• " + "\n• ".join(feas_errors)
                }), 400

        else:
            # ── GET: demo problem with time windows ────────────────
            # Demonstrates the new window feature:
            #   O'Hare accepts deliveries only 10:00am-12:00pm  (window 120-240)
            #   Willis Tower accepts deliveries only 9:00am-1:00pm (window 60-300)
            #   Navy Pier pickups only after 2:00pm (window 360-600)
            #   Schaumburg pickups only after 1:00pm (window 300-600)
            log.info("Request %s | GET demo problem (with time windows)", request_id)
            loc_names   = ["Depot (Addison, IL)", "O'Hare Airport", "Willis Tower", "Navy Pier", "Schaumburg"]
            loc_coords  = [(41.9314, -88.0126), (41.9742, -87.9073), (41.8789, -87.6359),
                           (41.8917, -87.6086), (42.0334, -88.0834)]
            loc_demands = [0, 10, 8, 6, 12]
            fleet       = [{"trailer": "53ft", "id": f"Truck-{i+1}"} for i in range(3)]
            loc_svc     = [0, 20, 20, 45, 45]
            loc_windows = [(0, 600), (120, 240), (60, 300), (360, 600), (300, 600)]
            n_del       = 2
            stop_zones  = ["depot", "airport", "commercial", "commercial", "industrial"]
            priorities  = [0, 1, 1, 1, 1]

        result, _, _ = run_solver(
            loc_names, loc_coords, loc_demands, fleet,
            loc_svc, loc_windows, n_del, stop_zones, priorities
        )

        if not result:
            msg = ("No solution found. Possible causes: "
                   "time windows are too tight to satisfy together, "
                   "not enough trucks for the total cargo, "
                   "zone restrictions prevent some stops from being reached, "
                   "or the delivery-before-pickup ordering cannot fit in the working day.")
            log.warning("Request %s | no solution | %s", request_id, msg)
            return jsonify({"error": msg}), 400

        log.info("Request %s | success | %d routes | %.1f miles",
                 request_id, len(result["routes"]), result["total_distance_miles"])
        return jsonify(result)

    except ValueError as e:
        log.warning("Request %s | validation error: %s", request_id, e)
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        log.error("Request %s | unexpected error: %s\n%s", request_id, e, traceback.format_exc())
        return jsonify({"error": f"Internal server error: {e}"}), 500


@app.route('/geometry', methods=['POST'])
def geometry():
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
