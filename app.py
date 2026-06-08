# ================================================================
# CutiGo: Flask API — app.py
# Deploy to Render (render.com)
#
# Endpoints:
#   GET  /              → health check
#   POST /recommend     → main itinerary endpoint
#   GET  /states        → list all available states
#   GET  /activities    → list all activity interests
#
# Folder structure on Render:
#   /
#   ├── app.py
#   ├── requirements.txt
#   ├── ml_model/
#   │   ├── cutigo_rf_model.pkl
#   │   ├── cutigo_encoders.pkl
#   │   └── cutigo_label_encoders.pkl
#   └── data/
#       └── cutigo_master_places.csv
# ================================================================

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ================================================================
# LOAD MODELS + DATA ON STARTUP
# ================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "ml_model")
DATA_DIR    = os.path.join(BASE_DIR, "data")

print("[CutiGo] Loading ML models...")
rf_model    = joblib.load(os.path.join(MODEL_DIR, "cutigo_rf_model.pkl"))
feature_enc = joblib.load(os.path.join(MODEL_DIR, "cutigo_encoders.pkl"))
label_enc   = joblib.load(os.path.join(MODEL_DIR, "cutigo_label_encoders.pkl"))
le_cat      = label_enc["category"]
print("[CutiGo] ML models loaded ✅")

print("[CutiGo] Loading places database...")
places = pd.read_csv(os.path.join(DATA_DIR, "cutigo_master_places.csv"))
if "place_name" in places.columns:
    places["display_name"] = places["place_name"]
elif "recommended_place" in places.columns:
    places["display_name"] = places["recommended_place"]
print(f"[CutiGo] {len(places):,} places loaded ✅")

# ================================================================
# CONSTANTS
# ================================================================
INPUT_FEATURES = [
    "state", "budget_preference", "activity_interest",
    "trip_duration", "group_type",
    "transportation_preference", "accommodation_preference",
]

DURATION_TO_DAYS = {
    "Half Day": 0.5,
    "1 Day":    1,
    "2-3 Days": 3,
    "4-7 Days": 5,
    "1 Week+":  7,
}

TIME_SLOTS = {
    2: ["Morning (9:00 AM – 12:00 PM)", "Afternoon (2:00 PM – 5:00 PM)"],
    3: ["Morning (9:00 AM – 12:00 PM)", "Afternoon (2:00 PM – 5:00 PM)",
        "Evening (7:00 PM – 9:00 PM)"],
}

VALID_STATES = sorted(places["state"].unique().tolist())

VALID_ACTIVITIES = [
    "Nature", "Sightseeing", "Culture", "Shopping",
    "Food", "Adventure", "Entertainment", "Relaxation"
]

VALID_BUDGETS       = ["Budget", "Moderate", "Premium", "Luxury"]
VALID_DURATIONS     = ["Half Day", "1 Day", "2-3 Days", "4-7 Days", "1 Week+"]
VALID_GROUPS        = ["Solo", "Couple", "Family", "Group of Friends"]
VALID_TRANSPORTS    = ["Car/Self-Drive", "Tour Bus", "Public Transport",
                       "Taxi/Grab", "Flight", "Boat"]
VALID_ACCOMMODATIONS = [
    "Hostel", "Budget Hotel", "Homestay", "Mid-range Hotel",
    "Boutique Hotel", "Resort/Luxury Hotel", "Glamping/Camp"
]

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def encode_input(user_prefs, activity_override=None):
    """Encode user preferences dict into numpy array."""
    prefs = user_prefs.copy()
    if activity_override:
        prefs["activity_interest"] = activity_override
    encoded = []
    for col in INPUT_FEATURES:
        le  = feature_enc[col]
        val = str(prefs.get(col, ""))
        encoded.append(int(le.transform([val])[0]) if val in le.classes_ else 0)
    return np.array(encoded).reshape(1, -1)


def predict_category(user_prefs, activity):
    """Run RF model for one activity, return top-3 category predictions."""
    arr      = encode_input(user_prefs, activity_override=activity)
    idx      = rf_model.predict(arr)[0]
    proba    = rf_model.predict_proba(arr)[0]
    cat      = le_cat.inverse_transform([idx])[0]
    top3_idx = np.argsort(proba)[::-1][:3]
    top3     = [
        {"category": le_cat.classes_[i], "confidence": round(float(proba[i]) * 100, 1)}
        for i in top3_idx
    ]
    return cat, top3


def get_more_info_link(place_row, place_name, state):
    """Return best available info link for a place."""
    booking = str(place_row.get("booking_link", ""))
    if booking and booking not in ("nan", "", "None", "NaN"):
        if not booking.startswith("http"):
            booking = "https://" + booking
        return booking, "website"
    # Fallback: Google Maps search
    query = place_name.replace(" ", "+") + "+" + state.replace(" ", "+")
    return f"https://www.google.com/maps/search/{query}", "google_maps"


def query_places_for_activity(user_state, predicted_cat, top3_cats,
                               n_needed, exclude_names=None):
    """
    Query places DB with fallback strategy:
    1. Same state + predicted category
    2. Same state + related categories
    3. Nationwide + predicted category
    """
    if exclude_names is None:
        exclude_names = set()

    all_cands = pd.DataFrame()

    # Priority 1: exact state + predicted category
    p1 = places[
        (places["state"] == user_state) &
        (places["category"] == predicted_cat) &
        (~places["display_name"].isin(exclude_names))
    ].copy()
    p1["match_quality"] = "exact"
    all_cands = pd.concat([all_cands, p1])

    # Priority 2: same state + related categories
    if len(all_cands) < n_needed:
        for item in top3_cats[1:]:
            cat = item["category"]
            p2  = places[
                (places["state"] == user_state) &
                (places["category"] == cat) &
                (~places["display_name"].isin(
                    set(all_cands["display_name"]) | exclude_names
                ))
            ].copy()
            p2["match_quality"] = "related"
            all_cands = pd.concat([all_cands, p2])
            if len(all_cands) >= n_needed * 2:
                break

    # Priority 3: nationwide fallback
    if len(all_cands) < n_needed:
        p3 = places[
            (places["category"] == predicted_cat) &
            (~places["display_name"].isin(
                set(all_cands["display_name"]) | exclude_names
            ))
        ].copy()
        p3["match_quality"] = "nationwide"
        all_cands = pd.concat([all_cands, p3])

    if len(all_cands) == 0:
        return pd.DataFrame()

    return (
        all_cands
        .drop_duplicates(subset=["display_name"])
        .sort_values("rating_imputed", ascending=False)
        .head(n_needed)
        .reset_index(drop=True)
    )


def split_days(total_days, activities):
    """Split days equally across activities."""
    n         = len(activities)
    base      = total_days // n
    remainder = total_days % n
    return {act: base + (1 if i < remainder else 0)
            for i, act in enumerate(activities)}


def get_ppd(total_days):
    """Places per day: 2 for short, 3 for longer trips."""
    return 2 if total_days <= 1 else 3


def validate_request(data):
    """Validate incoming request fields. Returns (is_valid, error_message)."""
    required = [
        "state", "budget_preference", "trip_duration",
        "group_type", "transportation_preference",
        "accommodation_preference", "activity_interest"
    ]
    for field in required:
        if field not in data:
            return False, f"Missing required field: '{field}'"

    if data["state"] not in VALID_STATES:
        return False, f"Invalid state. Choose from: {VALID_STATES}"

    if data["budget_preference"] not in VALID_BUDGETS:
        return False, f"Invalid budget. Choose from: {VALID_BUDGETS}"

    if data["trip_duration"] not in VALID_DURATIONS:
        return False, f"Invalid duration. Choose from: {VALID_DURATIONS}"

    if data["group_type"] not in VALID_GROUPS:
        return False, f"Invalid group_type. Choose from: {VALID_GROUPS}"

    if data["transportation_preference"] not in VALID_TRANSPORTS:
        return False, f"Invalid transportation. Choose from: {VALID_TRANSPORTS}"

    if data["accommodation_preference"] not in VALID_ACCOMMODATIONS:
        return False, f"Invalid accommodation. Choose from: {VALID_ACCOMMODATIONS}"

    # activity_interest can be string or list
    activities = data["activity_interest"]
    if isinstance(activities, str):
        activities = [a.strip() for a in activities.split(",")]
    for act in activities:
        if act not in VALID_ACTIVITIES:
            return False, f"Invalid activity: '{act}'. Choose from: {VALID_ACTIVITIES}"

    return True, ""


# ================================================================
# API ENDPOINTS
# ================================================================

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status":  "ok",
        "app":     "CutiGo Trip Matching API",
        "version": "1.0",
        "places":  len(places),
        "states":  len(VALID_STATES),
    })


@app.route("/states", methods=["GET"])
def get_states():
    """Return all available states."""
    return jsonify({
        "status": "ok",
        "states": VALID_STATES
    })


@app.route("/activities", methods=["GET"])
def get_activities():
    """Return all available activity interests."""
    return jsonify({
        "status":     "ok",
        "activities": VALID_ACTIVITIES
    })


@app.route("/options", methods=["GET"])
def get_options():
    """Return all valid options for every preference field."""
    return jsonify({
        "status": "ok",
        "options": {
            "states":           VALID_STATES,
            "activities":       VALID_ACTIVITIES,
            "budgets":          VALID_BUDGETS,
            "durations":        VALID_DURATIONS,
            "groups":           VALID_GROUPS,
            "transports":       VALID_TRANSPORTS,
            "accommodations":   VALID_ACCOMMODATIONS,
        }
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Main recommendation endpoint.

    Request body (JSON):
    {
        "state":                     "Sabah",
        "budget_preference":         "Moderate",
        "activity_interest":         ["Nature", "Adventure"],
        "trip_duration":             "4-7 Days",
        "group_type":                "Family",
        "transportation_preference": "Flight",
        "accommodation_preference":  "Mid-range Hotel"
    }

    Response (JSON):
    {
        "status": "ok",
        "user_preferences": { ... },
        "ai_predictions": [ ... ],
        "trip_summary": { ... },
        "itinerary": [
            {
                "day": 1,
                "day_label": "DAY 1",
                "activity": "Nature",
                "predicted_category": "Nature & Outdoors",
                "places": [
                    {
                        "time_slot": "Morning (9:00 AM – 12:00 PM)",
                        "place_name": "Kinabalu Park",
                        "category": "Nature & Outdoors",
                        "rating": 4.8,
                        "more_info_url": "https://...",
                        "link_type": "website",
                        "match_quality": "exact",
                        "state": "Sabah"
                    },
                    ...
                ]
            },
            ...
        ]
    }
    """
    # ── Parse request ─────────────────────────────────────────────
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No JSON body received"}), 400

    # ── Validate ──────────────────────────────────────────────────
    is_valid, err = validate_request(data)
    if not is_valid:
        return jsonify({"status": "error", "message": err}), 400

    # ── Normalise activities ──────────────────────────────────────
    raw_activities = data["activity_interest"]
    if isinstance(raw_activities, str):
        activities = [a.strip() for a in raw_activities.split(",") if a.strip()]
    else:
        activities = [a.strip() for a in raw_activities if a.strip()]
    activities = [a for a in activities if a in VALID_ACTIVITIES]
    if not activities:
        activities = ["Nature"]

    user_prefs = {
        "state":                     data["state"],
        "budget_preference":         data["budget_preference"],
        "activity_interest":         activities[0],  # for encoding
        "trip_duration":             data["trip_duration"],
        "group_type":                data["group_type"],
        "transportation_preference": data["transportation_preference"],
        "accommodation_preference":  data["accommodation_preference"],
    }

    # ── Days calculation ──────────────────────────────────────────
    duration   = data["trip_duration"]
    total_days = DURATION_TO_DAYS.get(duration, 1)
    is_half    = (total_days == 0.5)
    total_int  = 1 if is_half else int(total_days)
    ppd        = get_ppd(total_days)

    # ── Day → activity mapping ────────────────────────────────────
    allocation = {}
    if is_half or total_int == 1:
        day_activity_map = {1: activities}
    else:
        allocation = split_days(total_int, activities)
        day_activity_map = {}
        day_num = 1
        for act, n_days in allocation.items():
            for _ in range(n_days):
                day_activity_map[day_num] = [act]
                day_num += 1

    # ── ML Predictions ────────────────────────────────────────────
    act_preds = {}
    ai_predictions = []
    for act in activities:
        cat, top3 = predict_category(user_prefs, act)
        act_preds[act] = {"category": cat, "top3": top3}
        ai_predictions.append({
            "activity":           act,
            "predicted_category": cat,
            "confidence":         top3[0]["confidence"],
            "top3":               top3,
        })

    # ── Query places ──────────────────────────────────────────────
    used       = set()
    act_places = {}
    for act in activities:
        cat   = act_preds[act]["category"]
        top3  = act_preds[act]["top3"]
        days_for_act = 1 if (is_half or total_int == 1) else allocation.get(act, 1)
        n_need = days_for_act * ppd
        matched = query_places_for_activity(
            user_state    = data["state"],
            predicted_cat = cat,
            top3_cats     = top3,
            n_needed      = n_need,
            exclude_names = used,
        )
        act_places[act] = matched
        used.update(matched["display_name"].tolist() if len(matched) > 0 else [])

    # ── Build itinerary JSON ──────────────────────────────────────
    itinerary  = []
    all_places = []
    act_ptr    = {act: 0 for act in activities}
    time_slots = TIME_SLOTS.get(ppd, TIME_SLOTS[3])

    for day_num in sorted(day_activity_map.keys()):
        day_acts  = day_activity_map[day_num]
        day_label = "HALF DAY" if is_half else f"DAY {day_num}"

        day_places_list = []

        if is_half or total_int == 1:
            # Multiple activities in one day
            slots_per_act = max(1, ppd // max(len(day_acts), 1))
            slot_idx = 0
            for act in day_acts:
                matched = act_places[act]
                ptr     = act_ptr[act]
                for _ in range(slots_per_act):
                    if ptr >= len(matched) or slot_idx >= len(time_slots):
                        break
                    p     = matched.iloc[ptr]
                    name  = str(p.get("display_name", ""))
                    state = str(p.get("state", data["state"]))
                    link, link_type = get_more_info_link(p, name, state)
                    place_obj = {
                        "time_slot":    time_slots[slot_idx],
                        "place_name":   name,
                        "category":     str(p.get("category", "")),
                        "rating":       round(float(p.get("rating_imputed", 0)), 1),
                        "more_info_url":link,
                        "link_type":    link_type,
                        "match_quality":str(p.get("match_quality", "")),
                        "state":        state,
                    }
                    day_places_list.append(place_obj)
                    all_places.append(place_obj)
                    ptr      += 1
                    slot_idx += 1
                act_ptr[act] = ptr
        else:
            act     = day_acts[0]
            matched = act_places[act]
            ptr     = act_ptr[act]
            for s_idx in range(ppd):
                if ptr >= len(matched):
                    break
                p     = matched.iloc[ptr]
                name  = str(p.get("display_name", ""))
                state = str(p.get("state", data["state"]))
                link, link_type = get_more_info_link(p, name, state)
                place_obj = {
                    "time_slot":    time_slots[s_idx] if s_idx < len(time_slots) else time_slots[-1],
                    "place_name":   name,
                    "category":     str(p.get("category", "")),
                    "rating":       round(float(p.get("rating_imputed", 0)), 1),
                    "more_info_url":link,
                    "link_type":    link_type,
                    "match_quality":str(p.get("match_quality", "")),
                    "state":        state,
                }
                day_places_list.append(place_obj)
                all_places.append(place_obj)
                ptr += 1
            act_ptr[act] = ptr

        itinerary.append({
            "day":                day_num,
            "day_label":          day_label,
            "activity":           day_acts[0] if len(day_acts) == 1 else "Mixed",
            "predicted_category": act_preds[day_acts[0]]["category"] if len(day_acts) == 1 else "Mixed",
            "places":             day_places_list,
        })

    # ── Trip summary ──────────────────────────────────────────────
    avg_rating = round(float(np.mean([p["rating"] for p in all_places])), 2) if all_places else 0.0
    day_alloc  = [
        {"activity": act, "days": n, "category": act_preds[act]["category"]}
        for act, n in allocation.items()
    ] if allocation else []

    trip_summary = {
        "state":          data["state"],
        "duration":       duration,
        "total_days":     total_int,
        "activities":     activities,
        "day_allocation": day_alloc,
        "total_places":   len(all_places),
        "avg_rating":     avg_rating,
        "group":          data["group_type"],
        "budget":         data["budget_preference"],
        "transport":      data["transportation_preference"],
        "accommodation":  data["accommodation_preference"],
    }

    # ── Return response ───────────────────────────────────────────
    return jsonify({
        "status":           "ok",
        "user_preferences": data,
        "ai_predictions":   ai_predictions,
        "trip_summary":     trip_summary,
        "itinerary":        itinerary,
    })


# ================================================================
# RUN
# ================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
