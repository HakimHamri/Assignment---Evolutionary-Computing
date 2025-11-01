# streamlit_tv_ga.py
import streamlit as st
import pandas as pd
import random
from typing import List, Dict, Tuple

st.set_page_config(page_title="TV Scheduling GA", layout="wide")

st.title("📺 TV Program Scheduling — Genetic Algorithm (GA)")

# -------------------------
# Utility: Read CSV & parse
# -------------------------
st.sidebar.header("1) Load ratings CSV")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (header: Program,6,7,...23 or Program,slot1,slot2,...)", type=["csv"]
)

def read_ratings_csv(file) -> Tuple[Dict[str, Dict[int, float]], List[int]]:
    """
    Read CSV where header is: Program,6,7,8,... or Program,slotA,slotB...
    Returns:
      - ratings_map: {program_name: {slot_int: rating, ...}, ...}
      - slot_list: [6,7,8,...] (int list in order)
    """
    df = pd.read_csv(file)
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least one program column plus one slot column.")
    header = list(df.columns)
    # header[0] expected "Program"
    slot_headers = header[1:]
    # try to parse slot headers to ints (if they are numeric like "6","7"...)
    slot_list = []
    for h in slot_headers:
        try:
            slot_list.append(int(h))
        except Exception:
            # fallback: use index-based slot ids 0..n-1 if header not integer
            slot_list = list(range(len(slot_headers)))
            break

    ratings_map = {}
    for _, row in df.iterrows():
        program = str(row[header[0]])
        program_ratings = {}
        for idx, slot in enumerate(slot_list):
            # use df values in order
            val = row[header[idx+1]]
            try:
                r = float(val)
            except Exception:
                r = 0.0
            program_ratings[slot] = r
        ratings_map[program] = program_ratings

    return ratings_map, slot_list, df

# If user didn't upload, show sample instructions and example CSV
if uploaded_file is None:
    st.info(
        "Upload a CSV with columns: Program,6,7,...23  (i.e. 'Program' then time slot columns)\n\n"
        "Example rows:\n\n"
        "News,3.2,3.5,4.1,...\n\n"
        "If you don't have a CSV, upload one here to run the GA."
    )
    st.stop()

try:
    ratings_map, slot_list, df_raw = read_ratings_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.subheader("Loaded Ratings (first 10 rows)")
st.dataframe(df_raw.head(10), use_container_width=True)

# -------------------------
# GA Parameters (Streamlit)
# -------------------------
st.sidebar.header("2) GA Parameters (global)")
GEN = st.sidebar.number_input("Generations (GEN)", min_value=1, value=100, step=10)
POP = st.sidebar.number_input("Population size (POP)", min_value=4, value=50, step=1)
EL_S = st.sidebar.number_input("Elitism size (EL_S)", min_value=0, value=2, step=1)

st.sidebar.write("---")
st.sidebar.markdown("**Crossover & Mutation ranges**")
st.sidebar.markdown("- Crossover rate: 0.0 — 0.95")
st.sidebar.markdown("- Mutation rate: 0.01 — 0.05 (assignment constraint)")

# We'll allow the user to set defaults for 3 trials but ensure ranges
def param_slider(label, default, minv, maxv, step, key):
    return st.sidebar.slider(label, min_value=minv, max_value=maxv, value=default, step=step, key=key)

st.sidebar.header("Trial defaults (you can override per trial below)")
default_co = 0.8
default_mut = 0.02  # NOTE: assignment conflict: original said default .2 but allowed range is .01-.05
st.sidebar.markdown(
    "Default MUT_R set to 0.02 to respect the allowed range 0.01–0.05. "
    "Change it if you need a different value."
)

# -------------------------
# Per-trial parameters (3 trials)
# -------------------------
st.header("3) Run 3 Trials (different CO_R / MUT_R)")
st.write("Set crossover & mutation per trial, then click **Run trials**. Results will be shown below.")

with st.expander("Trial 1 parameters (default)"):
    t1_co = st.slider("Trial 1 — CO_R", 0.0, 0.95, default_co, 0.01, key="t1_co")
    t1_mut = st.slider("Trial 1 — MUT_R", 0.01, 0.05, default_mut, 0.01, key="t1_mut")

with st.expander("Trial 2 parameters"):
    t2_co = st.slider("Trial 2 — CO_R", 0.0, 0.95, default_co, 0.01, key="t2_co")
    t2_mut = st.slider("Trial 2 — MUT_R", 0.01, 0.05, default_mut, 0.01, key="t2_mut")

with st.expander("Trial 3 parameters"):
    t3_co = st.slider("Trial 3 — CO_R", 0.0, 0.95, default_co, 0.01, key="t3_co")
    t3_mut = st.slider("Trial 3 — MUT_R", 0.01, 0.05, default_mut, 0.01, key="t3_mut")

run_button = st.button("▶️ Run 3 Trials")

# -------------------------
# GA Implementation
# -------------------------
def fitness_of_schedule(schedule: List[str], ratings_map: Dict[str, Dict[int, float]], slot_list: List[int]) -> float:
    """Sum of ratings for schedule: schedule[i] is program assigned to slot_list[i]."""
    total = 0.0
    for i, program in enumerate(schedule):
        slot = slot_list[i]
        # If program doesn't have rating for slot (shouldn't happen), treat as 0
        total += ratings_map.get(program, {}).get(slot, 0.0)
    return total

def create_initial_population(programs: List[str], slot_count: int, pop_size: int) -> List[List[str]]:
    """Create a population of schedules (permutations). If number of programs >= slots -> use permutations without duplicates.
       If programs < slots -> allow some repeats (fill randomly)."""
    population = []
    for _ in range(pop_size):
        if len(programs) >= slot_count:
            # choose a random permutation of programs, take first slot_count
            perm = random.sample(programs, k=slot_count)
        else:
            # not enough unique programs: allow repeats, but try to diversify
            perm = []
            for _ in range(slot_count):
                perm.append(random.choice(programs))
        population.append(perm)
    return population

def tournament_selection(pop: List[List[str]], fitnesses: List[float], k=3) -> List[str]:
    participants = random.sample(list(range(len(pop))), k=min(k, len(pop)))
    best_idx = max(participants, key=lambda i: fitnesses[i])
    return pop[best_idx]

def ordered_crossover(parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
    """Ordered crossover (OX) — preserves order and avoids duplicates for permutations."""
    size = len(parent1)
    if size < 2:
        return parent1[:], parent2[:]
    a, b = sorted(random.sample(range(size), 2))
    def ox(p1, p2):
        child = [None] * size
        # copy segment
        child[a:b+1] = p1[a:b+1]
        # fill remaining from p2 in order
        p2_iter = [g for g in p2 if g not in child[a:b+1]]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = p2_iter[idx]
                idx += 1
        return child
    return ox(parent1, parent2), ox(parent2, parent1)

def swap_mutation(schedule: List[str]) -> List[str]:
    """Swap two positions — keeps permutation property."""
    if len(schedule) < 2:
        return schedule
    a, b = random.sample(range(len(schedule)), 2)
    schedule[a], schedule[b] = schedule[b], schedule[a]
    return schedule

def run_ga(ratings_map, slot_list, generations, pop_size, co_r, mut_r, elitism_size):
    programs = list(ratings_map.keys())
    slot_count = len(slot_list)

    pop = create_initial_population(programs, slot_count, pop_size)
    best_overall = None
    best_fitness = -float('inf')

    for gen in range(generations):
        fitnesses = [fitness_of_schedule(s, ratings_map, slot_list) for s in pop]
        # track best
        gen_best_idx = max(range(len(pop)), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_overall = pop[gen_best_idx].copy()

        # create next generation
        new_pop = []
        # elitism
        sorted_idx = sorted(range(len(pop)), key=lambda i: fitnesses[i], reverse=True)
        for i in range(min(elitism_size, len(pop))):
            new_pop.append(pop[sorted_idx[i]].copy())

        # fill rest
        while len(new_pop) < pop_size:
            parent1 = tournament_selection(pop, fitnesses)
            parent2 = tournament_selection(pop, fitnesses)
            if random.random() < co_r:
                child1, child2 = ordered_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mut_r:
                child1 = swap_mutation(child1)
            if random.random() < mut_r:
                child2 = swap_mutation(child2)

            new_pop.append(child1)
            if len(new_pop) < pop_size:
                new_pop.append(child2)

        pop = new_pop

    # final best_by_fitness
    final_fitnesses = [fitness_of_schedule(s, ratings_map, slot_list) for s in pop]
    best_idx = max(range(len(pop)), key=lambda i: final_fitnesses[i])
    return pop[best_idx], final_fitnesses[best_idx]

# -------------------------
# Run trials and show output
# -------------------------
def schedule_to_df(schedule: List[str], slot_list: List[int], ratings_map: Dict[str, Dict[int, float]]):
    rows = []
    for i, program in enumerate(schedule):
        slot = slot_list[i]
        rating = ratings_map.get(program, {}).get(slot, 0.0)
        rows.append({"Time": f"{slot:02d}:00", "Program": program, "Rating": rating})
    return pd.DataFrame(rows)

if run_button:
    st.write("Running trials... (this may take a few moments for large populations/generations)")
    trials = [
        ("Trial 1", t1_co, t1_mut),
        ("Trial 2", t2_co, t2_mut),
        ("Trial 3", t3_co, t3_mut),
    ]
    results = []
    for name, co_r, mut_r in trials:
        st.markdown(f"### {name} — CO_R={co_r:.2f}, MUT_R={mut_r:.3f}")
        best_schedule, best_fit = run_ga(
            ratings_map=ratings_map,
            slot_list=slot_list,
            generations=int(GEN),
            pop_size=int(POP),
            co_r=float(co_r),
            mut_r=float(mut_r),
            elitism_size=int(EL_S),
        )
        df_sched = schedule_to_df(best_schedule, slot_list, ratings_map)
        st.dataframe(df_sched, use_container_width=True)
        st.write(f"Total Ratings (fitness): **{best_fit:.4f}**")
        results.append((name, co_r, mut_r, best_fit, df_sched))
else:
    st.info("Set parameters and click 'Run 3 Trials' to execute the GA.")

# -------------------------
# Notes for your report
# -------------------------
st.header("Notes / What I changed & why")
st.markdown(
    """
- **CSV parsing:** I read the header to detect time-slot columns (e.g. 6..23). This avoids mis-indexing.
- **Fitness function:** now maps schedule position `i` → `slot_list[i]` so ratings are matched to the correct time slot.
- **No brute-force permutations:** I removed the factorial `initialize_pop` recursion. For realistic dataset sizes we use randomized population initialization and GA search.
- **Crossover & mutation:** I used ordered crossover (OX) and swap mutation to keep schedules as permutations and avoid duplicate programs in one schedule (when #programs >= #slots).
- **When programs < slots:** repeats are allowed (fallback). If you want to *force* uniqueness, ensure your CSV has at least `len(slot_list)` distinct programs.
- **Mutation default:** assignment text had a conflict (default 0.2 vs allowed 0.01-0.05). I set default MUT_R to **0.02** to remain inside the specified allowed range.
- **Report suggestion:** When you modify ratings in the CSV, keep a copy of the original and list row-by-row which values you changed and why (e.g., "Program X rating at 20:00 changed from 2.0 → 3.5 to simulate prime-time preference").
"""
)

st.markdown("If you want, I can also:")
st.markdown("- Add a download button to export each trial's schedule as CSV.")
st.markdown("- Run the algorithm now on your uploaded CSV and show the three trial outputs here (I already do that when you press Run).")
