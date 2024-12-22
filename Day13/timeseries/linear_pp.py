from pulp import *

model = LpProblem("CarProductionProblem", LpMaximize)

car_A = LpVariable("A",0, None,LpInteger)
car_B = LpVariable("B",0,None,LpInteger)

model += 30000 * car_A + 45000 * car_B,"Profit"

model += 3 * car_A + 4 * car_B <= 30, "Material Constraint 1"
model += 5 * car_A + 6 * car_B <= 60,   "Material Constraint 2"
model += 1.5 * car_A + 3 * car_B <= 21, "Labor Constraint "

#minimum production of car A
model += car_A >= 5, "MINImum A production"
model += car_B >= 3, "Minimum B production"
#the problem is solved using PuLP choice of solver
model.solve()

for v in model.variables():
    print(v.name,"=",v.varValue)

print("Total Profit:",value(model.objective)) #total profit
