# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import pandas as pd

np.random.seed(42)


# %%

def check_PF(priority, preced_list):
    preced_feasible = True
    for index in range(len(priority) - 1):
        base_item = priority[index]
        for succ_item in priority[index + 1:]:
            if succ_item in preced_list[base_item]:
                preced_feasible = False
    return preced_feasible


# %%

##test inputs
##Task:duration
T = {
    0: 1,
    1: 2,
    2: 1,
    3: 1,
    4: 3,
    5: 1,
    6: 3,
    7: 5,
}
##Task:predecessor
L = {1: [0], 2: [0], 3: [1], 4: [2], 5: [3, 4], 'B1': [3], 'B2': [4]}

# MileStone
B = {'B1': 100, 'B2': 200}

# Task:Resource
# R = {
#     0: 1,
#     1: 2,
#     2: 4,
#     3: 4,
# }
# Task:{resource:value}
R = {
    0: {0: 1, 1: 2},
    1: {0: 2, 1: 3},
    2: {0: 1, 1: 2},
    3: {0: 2, 1: 4},
}
# resource:max_available value
max_R = {
    0: 5,
    1: 5
}
all_tasks = set(T.keys())
predecessor_tasks = set(L.keys())
tasks_without_predecessors = all_tasks - {task for task, pred in L.items() if pred}
print('tasks without predecessors',tasks_without_predecessors)
successor_tasks = {task for predecessors in L.values() for task in predecessors}
tasks_without_successors = all_tasks - successor_tasks
print('tasks_without_successors',tasks_without_successors)
T['start'] = 0
T['end'] = 0
L['end'] = tasks_without_successors

for t in tasks_without_predecessors:
    L[t] = ['start']
print(L)
print(T)
default_resource_value = {k: 0 for k in max_R.keys()}
print('default_resource_value',default_resource_value)
for task in all_tasks - {'start', 'end'}:
    if task not in R:
        R[task] = default_resource_value
R['start'] = default_resource_value
R['end'] = default_resource_value

print('R',R)

# %%
task_weights = {task: 0 for task in T}
print('task_weights',task_weights)

# Function to calculate task weights recursively
def calculate_task_weights(task, milestone):
    # Calculate the weight based on milestone weight and dependent tasks
    weight = B.get(milestone, 0)
    for dep_task in L.get(task, []):
        task_weights[dep_task] = max(calculate_task_weights(dep_task, milestone), task_weights.get(dep_task, 0))

    # Update the weight in the task_weights dictionary
    task_weights[task] = weight
    return weight


for milestone in B:
    calculate_task_weights(milestone, milestone)
print('task_weights',task_weights)
# %%
priority = []


def schedule_tasks():
    schedule = []
    remaining_tasks = set(T.keys())

    while remaining_tasks:
        schedulable_tasks = [task for task in remaining_tasks if all(dep in schedule for dep in L.get(task, []))]
        print('schedulable_tasks',schedulable_tasks)
        if not schedulable_tasks:
            print("inside")
            # There are unschedulable tasks, choose the ones with the highest weight
            max_weight = max(task_weights[task] for task in remaining_tasks)
            schedulable_tasks = [task for task in remaining_tasks if task_weights[task] == max_weight]

        next_task = max(schedulable_tasks, key=lambda x: (task_weights[x], x))
        print('next_task',next_task)
        schedule.append(next_task)
        remaining_tasks.remove(next_task)

    return list(set(schedule) - {'start', 'end'})


# Start scheduling tasks
priority = schedule_tasks()
print('priority',priority)

def plot_gantt(finishes, T, title):
    today = datetime.today()

    sorted_keys = sorted(T.keys(), key=finishes.get)
    df = pd.DataFrame({
        'task': sorted_keys,
        'duration': [T[key] for key in sorted_keys]
    }, columns=['task', 'duration', 'start', 'finish', 'Fj'])
    df['Fj'] = df['task'].apply(lambda x: finishes.get(x, 0))
    for index, row in df.iterrows():
        finish_time = row['Fj']
        duration = row['duration']
        df.at[index, 'finish'] = today + timedelta(days=finish_time)
        df.at[index, 'start'] = today + timedelta(days=(finish_time - duration))
    df['start'] = pd.to_datetime(df['start']).dt.normalize()
    df['finish'] = pd.to_datetime(df['finish']).dt.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    print(df)
    start_index = df[df['task'] == 'start'].index[0]
    end_index = df[df['task'] == 'end'].index[0]
    df.at[start_index, 'start'] = today + timedelta(days=-1)
    df.at[start_index, 'finish'] = today + timedelta(days=-1)
    df.at[end_index, 'start'] = today + timedelta(days=1)
    df.at[end_index, 'finish'] = today + timedelta(days=1)
    df['task_duration'] = df['task'].astype(str) + " (" + df['duration'].astype(str) + ")"
    fig = px.timeline(df, x_start="start", x_end="finish", y="task", color="task",
                      labels={"task": "Task"},
                      title=title,
                      template="plotly",
                      text="task_duration")
    fig.update_yaxes(categoryorder="total ascending")

    # Show figure
    fig.show()


# %%

def PSGS(T, L, R, P):
    # T => Task:duration
    # L => Task:predecessor
    # R => Task:{resource:value}
    # P => priority list
    priority_list = P.copy()  # we dont want to mess with the original priority list
    priority_list.insert(0, 'start')
    priority_list.append('end')
    ##initialization

    n = len(T.keys())  ## total number of activities

    # def minAg(Ag, g):
    #     min_task = min(Ag[g], key=Fj.get)
    #     return min_task, Fj[min_task]

    # Step 1
    g = 0
    t = {g: 0}  ## interation time
    A = {g: {'start'}}  ## activities in progress at tg
    C = {g: []}  ## activities completed at tg
    D = {g: set()}  ##feasible activities at tg
    Rk = {g: max_R}

    Fj = {
        'start': 0
    }

    while len(A[g]) + len(C[g]) <= n:
        g = g + 1
        A[g] = A[g - 1].copy()
        C[g] = C[g - 1].copy()
        Rk[g] = Rk[g - 1].copy()
        D[g] = D[g - 1].copy()
        print('A',A[g])
        print('C',C[g])
        print('D',D[g])
        print('Rk',Rk[g])


        # completed_activity, t[g] = minAg(A, g)
        completed_activity = priority_list.pop(0)
        t[g] = min(Fj[t] for t in A[g])

        C[g].append(completed_activity)

        A[g].difference_update({completed_activity})

        for r in R[completed_activity].keys():
            Rk[g][r] = Rk[g - 1][r] + R[completed_activity][r]

        D[g].update([l for l in L.keys() if set(L[l]).issubset(C[g]) and l not in C[g] and l not in A[g] and l in T])

        if len(D[g]) > 0:
            for d in D[g]:
                Fj[d] = t[g] + T[d]

    # Select tasks from new_tasks based on resource availability to add to Ag[g]
        for task in D[g]:
            if all(R[task][r] <= Rk[g][r] for r in R[task].keys()):
                A[g].add(task)
                Rk[g] = {r: Rk[g][r] - R[task][r] for r in R[task].keys()}
        D[g].difference_update(A[g])
        if len(A[g]) == 0:
            break
    print(f'Solved in {g} iterations')
    print(C[g])
    return C[g], Fj


sequence, finishes = PSGS(T, L, R, priority)

sequence.remove('start')
sequence.remove('end')
check_PF(sequence, L)



# %%
plot_gantt(finishes, T, "Tasks scheduling PSGS")




# %%
print("SGS")
def SGS(T, L, R, P):
    priority_list = P.copy()
    priority_list.insert(0, 'start')
    priority_list.append('end')
    n = len(T.keys())  ## total number of activities
    # Step 1
    F = {'start': 0}
    Sg = {0: ['start']}
    Dg = {}
    Rk = {t: max_R for t in range(sum(T.values()))}

    def calculate_Dg(Cg, L, T):
        D = set()
        for task in T.keys():
            if task not in Cg and (not L.get(task) or set(L[task]).issubset(Cg)):
                D.add(task)
        return D

    for g in range(1, n + 1):

        Sg[g] = Sg[g - 1].copy()
        print('Sg',Sg[g])
        # Calculate Dg:
        Dg[g] = calculate_Dg(Sg[g], L, T)
        print('Dg',Dg[g])
        if Dg[g]:
            j = priority_list.pop(0)
            if j in Sg[g]:
                continue

            if j in Dg[g]:
                # for j in Dg[g]:
                EF = max(F.get(h, 0) + T[j] for h in L[j])
                while not all(R[j][r] <= Rk[t][r] for t in
                              range(EF - T[j],
                                    EF) for r in R[t].keys()):
                    EF += 1
                F[j] = EF  # Update the finish time of task j
                Sg[g].append(j)
                Dg[g].remove(j)
                for t in range(EF - T[j], EF):
                    Rk[t] = {r: Rk[t][r] - R[j][r] for r in R[j].keys()}  # Update the resources
                # break
            else:
                raise Exception('error: the priorority list is not feasible')

        else:
            print('error: no feasible tasks found')
            break

    print(sorted(Sg[n], key=lambda task: F.get(task, 0)))
    return sorted(Sg[n], key=lambda task: F.get(task, 0)), F 


sequence, finishes = SGS(T, L, R, priority)
print('finishes',finishes)
sequence.remove('start')
sequence.remove('end')
check_PF(sequence, L)



# %%

plot_gantt(finishes, T, "Tasks scheduling SSGS")


