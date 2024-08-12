import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np

np.random.seed(42)

def check_PF(priority,preced_list):
  preced_feasible = True
  for index in range(len(priority)-1):
    base_item = priority[index]
    for succ_item in priority[index+1:]:
      if succ_item in preced_list[base_item]:
        preced_feasible = False
  return preced_feasible

preced_list = {0:[],1:[0],2:[1],3:[1],4:[3], 5:[3],6:[2,5], 7:[2,5,4], 8:[7], 9:[8,6]}
priority = [0,1,2,3,5,4,7,6,8,9]
check_PF(priority, preced_list)


def all_predecessors(activity_index,preced_list):
  pred_list = []
  if preced_list[activity_index] == []:
    return list(set(pred_list))
  else:
    pred_list += preced_list[activity_index]
    for item in preced_list[activity_index]:
      pred_list += all_predecessors(item,preced_list)
  return list(set(pred_list))

all_predecessors(4,preced_list)


##test inputs
##Task:duration
T = {
    0: 1,
    1: 2,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7:5
}
##Task:predecessor
L = {1: [0], 2: [0], 3: [2, 1]}

##Task:Resource
R = {
    0: 1,
    1: 2,
    2: 4,
    3: 4,
}


max_R = 5

all_tasks = set(T.keys())
print('all_tasks',all_tasks)
predecessor_tasks = set(L.keys())
print('predecessor_tasks',predecessor_tasks)
tasks_without_predecessors = all_tasks - {task for task, pred in L.items() if pred}
print('tasks_without_predecessors',tasks_without_predecessors)
successor_tasks = {task for predecessors in L.values() for task in predecessors}
print('successor_tasks',successor_tasks)
tasks_without_successors = all_tasks - successor_tasks
print('tasks_without_successors',tasks_without_successors)
T['start'] = 0
T['end'] = 0
L['end'] = tasks_without_successors
R['start']=0
R['end']=0
for t in tasks_without_predecessors:
    L[t] = ['start']
print(L)
print(T)
default_resource_value=1
for task in all_tasks - {'start', 'end'}:
    if task not in R:
        R[task] = default_resource_value

print(R)
# def PSGS(T,L,R):
#     ##initialization


#     n = len(T.keys())  ## total number of activities

#     def minAg(Ag, g):
#         min_task = min(Ag[g], key=Fj.get)
#         return min_task, Fj[min_task]

#     # Step 1
#     g = 0
#     tg = {g: 0}  ## interation time
#     Ag = {g: {'start'}}  ## activities in progress at tg
#     Cg = {g: []}  ## activities completed at tg
#     Dg = {g: set()}  ##feasible activities at tg
#     Rk = {g: max_R}

#     Fj = {
#         'start': 0
#     }

#     while len(Ag[g]) + len(Cg[g]) <= n:
#         g = g + 1
#         Ag[g] = Ag[g - 1].copy()
#         Cg[g] = Cg[g - 1].copy()
#         Rk[g] = Rk[g - 1]
#         Dg[g] = Dg[g - 1].copy()

#         completed_activity, tg[g] = minAg(Ag, g)

#         Cg[g].append(completed_activity)

#         Ag[g].difference_update({completed_activity})
#         Rk[g] = Rk[g - 1] + R[completed_activity] ## or we can do max_r-sum(R[t] for t in Ag[g])
#         Dg[g].update([l for l in L.keys() if set(L[l]).issubset(Cg[g]) and l not in Cg[g] and l not in Ag[g]])

#         if len(Dg[g]) > 0:
#             for d in Dg[g]:
#                 Fj[d] = tg[g] + T[d]

#         # Select tasks from new_tasks based on resource availability to add to Ag[g]
#         for task in sorted(Dg[g], key=R.get):  # sort tasks by resource requirement
#             if R[task] <= Rk[g]:
#                 Ag[g].add(task)
#                 Rk[g] -= R[task]
#         Dg[g].difference_update(Ag[g])
#         if len(Ag[g]) == 0:
#             break
#     print(f'Solved in {g} iterations')
#     print(Cg[g])
#     return Cg[g],Fj

# sequence,finishes=PSGS(T,L,R)

# def plotTasks(F,T,title='Scheduling Tasks'):
#     # Initialize the figure and subplot
#     fig, ax = plt.subplots()
#     # Set the y-limits according to the max resources
#     ax.set_ylim(0, max_R)
#     # Set the x-limits according to the max finish time
#     f_max = max(F.values())
#     ax.set_xlim(0, f_max)

#     # Create a colormap, generate random colors
#     # colormap = {task: mcolors.to_rgba(np.random.rand(4)) for task in Cg[g]}
#     colormap = {task: mcolors.to_rgba(np.random.rand(4)) for task in T.keys()}
#     running_resources = [0] * (max(F.values()) + 1)
#     # Construct plot
#     for task in T.keys():
#         if task in {'start', 'end'}:
#             continue

#         # Retrieve data for each task
#         start = F[task] - T[task]
#         finish = F[task]
#         resources = R[task]
#         h = running_resources[start]
#         # Create a rectangle (patch) for each task (box on plot)
#         rect = patches.Rectangle((start, h), finish - start, resources, facecolor=colormap[task])
#         ax.add_patch(rect)

#         # Add a text label for each task
#         ax.text(start + (finish - start) / 2, h + resources / 2, f'Task {task}', ha='center', va='center',
#                 color='white')
#         for i in range(start, finish):
#             running_resources[i] += resources
#     # Set labels and title:

#     # Set labels and title
#     ax.set_xlabel("Time (days)")
#     ax.set_ylabel("Resources")
#     ax.set_title(title)
#     plt.show()

def plotTasks(F, T, title):
    def create_schedule_matrix(T, R, F, max_R):

        f_max = max(F.values())
        schedule_matrix = np.zeros((f_max + 1, max_R))

        # Walk through each task
        for task in T.keys():
            if task in {'start', 'end'}:
                continue

            start = F[task] - T[task]
            end = F[task]
            resources = R[task]

            # Find the minimum available level to put the task
            for level in range(max_R):
                if (schedule_matrix[start:end, level:level + resources] == 0).all():  # Check if the level is free
                    schedule_matrix[start:end, level:level + resources] = task  # If free, assign the task
                    break  # Go to the next task once a level has been found

        return schedule_matrix
    schedule_matrix = create_schedule_matrix(T, R, finishes, max_R)
    fig, ax = plt.subplots()
    # Set the y-limits according to the max resources
    ax.set_ylim(0, max_R)
    # Set the x-limits according to the max finish time
    f_max = max(F.values())
    ax.set_xlim(0, f_max)
    colormap = {task: mcolors.to_rgba(np.random.rand(4)) for task in T.keys()}
    for task in T.keys():
        if task in {'start', 'end'}:
            continue

        # Retrieve data for each task
        start = F[task] - T[task]
        finish = F[task]
        resources = R[task]

        # Find the task's level in the schedule matrix
        h = np.where(schedule_matrix[start, :] == task)[0][0]

        rect = patches.Rectangle((start, h), finish - start, resources, facecolor=colormap[task])
        ax.add_patch(rect)

        ax.text(start + (finish - start) / 2, h + resources / 2, f'Task {task}', ha='center', va='center',
                color='white')

    # Set labels and title
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Resources")
    ax.set_title(title)
    plt.show()

# plotTasks(finishes,T,"Tasks scheduling PSGS")

# sequence.remove('start')
# sequence.remove('end')
# check_PF(sequence, L)

def SGS(T, L, R):
  max_R = 5
  n = len(T.keys())  ## total number of activities
  # Step 1
  g = 0
  F = {0: 0}
  Sg = {0: ['start']}
  Dg = {}
  Rk = {t: max_R for t in range(sum(T.values()))}


  def calculate_Dg(Cg, L, T):
      D = set()
      for task in T.keys():
          if task not in Cg and set(L[task]).issubset(Cg):
              D.add(task)
      return D
  
  for g in range(1, n):
      print('Sg',Sg)  
      Sg[g] = Sg[g - 1].copy()
      # Calculate Dg:
      Dg[g] = calculate_Dg(Sg[g], L, T)
      print('Dg',Dg)
      if Dg[g]:
          for j in Dg[g]:
              EF = max(F.get(h, 0) + T[j] for h in L[j])
              while not all(R[j] <= Rk[t] for t in
                            range(EF - T[j], EF)):  ## not sure on this point as this should belong to [EFj - Pi,LFj - Pi]
                  EF += 1
              F[j] = EF  # Update the finish time of task j
              Sg[g].append(j)
              Dg[g].remove(j)
              for t in range(EF - T[j], EF):
                  Rk[t] -= R[j]  # Update the resources
              break

      else:
          print('error: no feasible tasks found')
          break

  print(sorted(Sg[n - 1], key=lambda task: F.get(task, 0)))
  return sorted(Sg[n - 1], key=lambda task: F.get(task, 0)),F

sequence,finishes=SGS(T,L,R)

plotTasks(finishes,T,"Tasks scheduling SSGS")

sequence.remove('start')
sequence.remove('end')
check_PF(sequence, L)