__author__ = "sion"

import sys

"""
    Tableau simplex algorithm to solve base linear program.
"""
class SimplexTable:
    def __init__(self, decision_var_count, constraints_count, lhs_matrix, obj_func_coefficients, rhs_vector):
        """
        Initialises the Simplex Table.

        :param decision_var_count: Number of decision variables.
        :param constraints_count:  Number of constraints.
        :param lhs_matrix: Coefficients of the constraint matrix (LHS).
        :param obj_func_coefficients:  Coefficients for the objective function.
        :param rhs_vector: RHS constraint vector.
        """
        self.decision_var_count = decision_var_count
        self.constraints_count = constraints_count
        self.obj_coefficients = obj_func_coefficients
        self.rhs = rhs_vector

        # Initial constraint matrix with slack variables
        self.constraints = [[0] * (decision_var_count + constraints_count) for _ in range(constraints_count)]
        for i in range(constraints_count):
            self.constraints[i][:decision_var_count] = lhs_matrix[i]
            self.constraints[i][decision_var_count + i] = 1  # Indicate the existence of slack variables

        # Initial c_j
        self.c_j = obj_func_coefficients + [0] * constraints_count

        # Initial values
        self.z_j = [0] * (decision_var_count + constraints_count)
        self.theta = [0] * constraints_count
        self.decision_vars = [0] * (decision_var_count + constraints_count)
        self.Z = 0  # The optimum Z we want to find

        # Fill RHS of the constraints
        self.basic_vars = [decision_var_count + i for i in range(constraints_count)]

        # Solve the linear problem
        self.solve()

    def solve(self):
        """
        Solve the linear program using the Simplex method.
        """
        while True:
            # Update z_j with the dot product
            self.calculate_z_j()

            # Check halt condition
            delta = [self.c_j[j] - self.z_j[j] for j in range(len(self.c_j))]
            if all(d <= 0 for d in delta):  # If all values of c_j - z_j are <= 0
                break  # Optimal solution is found

            # Find pivot element
            entering, exiting = self.find_pivot(delta)

            # Perform pivot operation
            self.pivot(entering, exiting)

        # The final decision variables
        final_decision_vars = []
        for i in range(self.constraints_count):
            # Output only if the variable is from original decision variables
            if self.basic_vars[i] < self.decision_var_count:
                self.decision_vars[self.basic_vars[i]] = self.rhs[i]
                if self.rhs[i] != 0:  # Filter zeros
                    final_decision_vars.append(int(self.rhs[i]))  # Convert to int

        # Calculate the final Z value from the objective function
        self.Z = sum(self.c_j[self.basic_vars[i]] * self.rhs[i] for i in range(self.constraints_count))

        self.output(final_decision_vars)

    def calculate_z_j(self):
        """
        Calculate z_j values (the dot product between the basic variables and the constraint matrix.)
        """
        # The dot product calculation
        for j in range(len(self.z_j)):
            self.z_j[j] = sum(self.c_j[self.basic_vars[i]] * self.constraints[i][j] for i in range(self.constraints_count))

    def find_pivot(self, delta):
        """
        Finds the entering and exiting variables.
        :param delta: The list of delta values.
        :return entering, exiting: The index of the entering variables.
        """
        # Find the entering variable (the largest positive c_j - z_j)
        entering = delta.index(max(delta))

        # Calculate theta for all rows
        self.calculate_theta(entering)
        # Select the minimum positive theta
        exiting = self.theta.index(min(self.theta))

        return entering, exiting

    def calculate_theta(self, pivot_col):
        """
        Calculate theta values.

        :param pivot_col: The column index corresponding to the entering variable.
        """
        # Avoid division by 0 or negative numbers
        for i in range(self.constraints_count):
            if self.constraints[i][pivot_col] > 0:
                # the ratio of RHS to the pivot column values.
                self.theta[i] = self.rhs[i] / self.constraints[i][pivot_col]
            else:
                self.theta[i] = float('inf')

    def pivot(self, entering, exiting):
        """
        Perform pivot operation to update the table.

        :param entering: The index of the entering variable (column).
        :param exiting: The index of the exiting variable (row).
        """
        pivot_value = self.constraints[exiting][entering]

        # Update the exiting row
        self.constraints[exiting] = [x / pivot_value for x in self.constraints[exiting]]
        self.rhs[exiting] /= pivot_value

        # Update other rows (the pivot operations)
        for i in range(self.constraints_count):
            if i != exiting:
                factor = self.constraints[i][entering]
                self.constraints[i] = [self.constraints[i][j] - factor * self.constraints[exiting][j] for j in
                                       range(len(self.constraints[i]))]
                self.rhs[i] -= factor * self.rhs[exiting]

        # Update the basic variables
        self.basic_vars[exiting] = entering

    def output(self, final_decision_vars):
        """
        Outputs the result. (Below is an example of output_q2.txt)

        # Optimal_Values_of_Decision_Variables
        5, 9
        # Optimal_Value_of_Objective_Function
        23
        """
        with open("output_q2.txt", "w+") as f:
            f.write("# Optimal_Values_of_Decision_Variables (n)\n")
            f.write(", ".join(map(str, final_decision_vars)) + "\n")
            f.write("# Optimal_Value_of_Objective_Function (e)\n")
            f.write(str(int(self.Z)) + "\n")  # convert to int


if __name__ == '__main__':
    # python q2.py <filename>
"""
# N_Decision_Variables
2
# N_Constraints
5
# Coefficients_of_Objective_Function
1,2
# Constraints_Matrix_LHS
4,1
3,2
2,3
0,1
-1,1
# constraints_Vector_RHS
44
39
37
9
6
"""
# Above is the input file format
    with open(sys.argv[1], "r") as file:
        # Read the input, ignoring comment lines
        data = [line.strip() for line in file if not line.startswith("#")]

        # Number of decision variables
        n_decision_vars = int(data[0].strip())

        # Number of constraints
        n_constraints = int(data[1].strip())

        # Coefficients of the objective function
        coefficients_of_objective_function = list(map(int, data[2].strip().split(',')))

        # LHS of constraints (matrix)
        lhs = [list(map(int, line.strip().split(','))) for line in data[3:3 + n_constraints]]

        # RHS of constraints (vector)
        rhs = list(map(int, data[3 + n_constraints:3 + 2 * n_constraints]))

        simplex = SimplexTable(n_decision_vars, n_constraints, lhs, coefficients_of_objective_function, rhs)
