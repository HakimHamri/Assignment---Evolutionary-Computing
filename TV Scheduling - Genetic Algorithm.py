# streamlit_tv_ga.py
import streamlit as st
import pandas as pd
import random
import io

st.set_page_config(page_title="TV Scheduling GA", layout="wide")

st.title("📺 TV Program Scheduling — Genetic Algorithm")
st.markdown(
    """
This app runs a Genetic Algorithm to produce a TV schedule.
- Upload a CSV of program ratings (one row per program).
- Columns: Program name, then rating for each time slot (e.g. 6,7,...,23).
"""
)

# ----------------------------
# Utilities: CSV loader
# ----------------------------
def read_csv_to_dict(file_buffer, expected_slots=None):
    df = pd.read_csv(file_buffer)
    if df.shape[1] < 2:
        raise ValueError("CSV must have program name + rating columns.")
    program_col = df.columns[0]
    programs = df[program_col].astype(str).tolist()
    rating_cols = df.columns[1:].tolist()
    # Convert to floats
    ratings = {}
    for idx, program in enumerate(programs):
        row_vals = df.iloc[idx, 1:].tolist()
        try:
            row_vals = [float(x) for x in row_vals]
        except Exception as e:
            raise ValueError(f"Non-numeric rating detected for program '{program}': {e}")
        if expected_slots is not None and len(row_vals) != expected_slots:
            raise ValueError(
                f"Program '{program}' has {len(row_vals)} ratings but expected {expected_slots} (one per time slot)."
            )
        ratings[program] = row_vals
    return ratings, rating_cols

# ----------------------------
# GA core (chromosome: list of program names length = T)
# ----------------------------
def fitness_function(schedule, ratings):
    # schedule: list of program names, length T
    total = 0.0
    for slot_idx, program in enumerate(schedule):
        total += ratings[program][slot_idx]
    return total

def initialize_population(all_programs, T, population_size):
    pop = []
    for _ in range(population_size):
        # allow repeats: choose a program for each timeslot
        chromosome = [random.choice(all_programs) for _ in range(T)]
        pop.append(chromosome)
    return pop

def one_point_crossover(p1, p2):
    if len(p1) < 2:
        return p1.copy(), p2.copy()
    point = random.randint(1, len(p1)-1)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2

def mutate(chromosome, all_programs):
    i = random.randrange(len(chromosome))
    chromosome[i] = random.choice(all_programs)
    return chromosome

def genetic_algorithm(
    ratings,
    all_programs,
    T,
    generations=100,
    population_size=50,
    crossover_rate=0.8,
    mutation_rate=0.02,
    elitism_size=2,
    seed=None,
):
    if seed is not None:
        random.seed(seed)

    # initialize
    population = initialize_population(all_programs, T, population_size)

    for gen in range(generations):
        # sort by fitness descending
        population.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
        new_pop = []
        # elitism
        new_pop.extend(population[:elitism_size])

        # create till population_size
        while len(new_pop) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = one_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1, all_programs)
            if random.random() < mutation_rate:
                child2 = mutate(child2, all_programs)

            new_pop.append(child1)
            if len(new_pop) < population_size:
                new_pop.append(child2)

        population = new_pop

    # return best individual
    population.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
    best = population[0]
    best_score = fitness_function(best, ratings)
    return best, best_score

# ----------------------------
# Streamlit UI
# ----------------------------
st.sidebar.header("Upload & Time Slots")
uploaded_file = st.sidebar.file_uploader("Upload program ratings CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample demo data", value=False)

# time slots: default 6..23 (18 slots)
start_hour = st.sidebar.number_input("Start hour (inclusive)", min_value=0, max_value=23, value=6)
end_hour = st.sidebar.number_input("End hour (inclusive)", min_value=0, max_value=23, value=23)
if end_hour <= start_hour:
    st.sidebar.error("End hour must be greater than start hour.")
all_time_slots = list(range(int(start_hour), int(end_hour) + 1))
T = len(all_time_slots)

ratings = {}
rating_cols = []

if use_sample and not uploaded_file:
    # build a tiny sample with 4 programs and T slots
    programs = ["News", "ShowA", "MovieA", "Sports"]
    data = {
        "Program": programs
    }
    for i, h in enumerate(all_time_slots):
        # simple synthetic ratings
        data[str(h)] = [round(random.uniform(0.5, 3.5) + (0.5 if h >= 19 else 0), 2) for _ in programs]
    df_sample = pd.DataFrame(data)
    st.info("Using generated sample data (you can still upload your own CSV).")
    st.dataframe(df_sample)
    buf = io.StringIO()
    df_sample.to_csv(buf, index=False)
    buf.seek(0)
    ratings, rating_cols = read_csv_to_dict(buf, expected_slots=T)
elif uploaded_file:
    try:
        ratings, rating_cols = read_csv_to_dict(uploaded_file, expected_slots=T)
        st.success(f"Loaded {len(ratings)} programs with {T} time slots ({all_time_slots[0]}–{all_time_slots[-1]}).")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
else:
    st.info("Upload a CSV or check 'Use sample demo data' to proceed.")
    st.stop()

all_programs = list(ratings.keys())

# GA parameter inputs
st.sidebar.header("GA parameters (default values)")
generations = st.sidebar.number_input("Generations", min_value=1, max_value=10000, value=100)
population_size = st.sidebar.number_input("Population size", min_value=2, max_value=1000, value=50)
elitism_size = st.sidebar.number_input("Elitism size", min_value=0, max_value=50, value=2)

# Trial controls: allow 1 or 3 trials
trials = st.sidebar.selectbox("Number of trials", options=[1, 3], index=1)

# Parameter controls for each trial
trial_params = []
st.sidebar.markdown("---")
st.sidebar.write("Set CO_R and MUT_R for each trial")
for i in range(trials):
    st.sidebar.markdown(f"**Trial {i+1}**")
    co_r = st.sidebar.slider(f"CO_R (trial {i+1})", min_value=0.0, max_value=0.95, value=0.8, step=0.01, key=f"co_{i}")
    # mutation range set by assignment: 0.01 - 0.05 (default 0.02)
    mut_r = st.sidebar.slider(f"MUT_R (trial {i+1})", min_value=0.01, max_value=0.05, value=0.02, step=0.01, key=f"mut_{i}")
    trial_params.append((co_r, mut_r))

run_seed = st.sidebar.number_input("Random seed (0 = random)", min_value=0, value=0)
st.sidebar.markdown("---")
run_button = st.sidebar.button("Run GA")

# Main app: run when pressed
if run_button:
    st.header("Results")
    cols = st.columns(trials)
    results = []

    for i in range(trials):
        with cols[i]:
            st.subheader(f"Trial {i+1}")
            co_r, mut_r = trial_params[i]
            seed = None if run_seed == 0 else int(run_seed + i)  # Vary seed per trial if provided

            best_schedule, best_score = genetic_algorithm(
                ratings=ratings,
                all_programs=all_programs,
                T=T,
                generations=int(generations),
                population_size=int(population_size),
                crossover_rate=float(co_r),
                mutation_rate=float(mut_r),
                elitism_size=int(elitism_size),
                seed=seed,
            )

            # prepare human-friendly table
            df_out = pd.DataFrame({
                "Hour": [f"{h:02d}:00" for h in all_time_slots],
                "Program": best_schedule,
                "Rating": [ratings[prog][idx] for idx, prog in enumerate(best_schedule)]
            })
            st.write(f"Parameters: CO_R={co_r}, MUT_R={mut_r}")
            st.dataframe(df_out, use_container_width=True)
            st.write(f"Total Rating (fitness): **{best_score:.4f}**")
            results.append((co_r, mut_r, best_schedule, best_score))

    # summary
    st.markdown("---")
    st.subheader("Summary of trials")
    summary_rows = []
    for idx, (co, mu, sched, score) in enumerate(results, start=1):
        summary_rows.append({
            "Trial": idx,
            "CO_R": co,
            "MUT_R": mu,
            "Total Rating": round(score, 4)
        })
    st.table(pd.DataFrame(summary_rows))

    st.success("GA runs finished.")
