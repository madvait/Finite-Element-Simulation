#disclosure: plot methods have been created with the help of gen ai
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

x = symbols('x')
y = symbols('y')
s = symbols('s')
t = symbols('t')
k = 0.056*Matrix([[1,0],[0,1]])

class Trielement:
    def __init__(self,x1,x2,x3, id1,id2,id3):
        self.x1 = x1[0]#float(x1[0])
        self.x2 = x2[0]#float(x2[0])
        self.x3 = x3[0]#float(x3[0])
        self.y1 = x1[1]#float(x1[1])
        self.y2 = x2[1]#float(x2[1])
        self.y3 = x3[1]#float(x3[1])
        self.id1 = id1
        self.id2 = id2
        self.id3 = id3
        self.node_id = [id1,id2,id3]
        self.node_set = [x1,x2,x3]#[x1.tolist(),x2.tolist(),x3.tolist()]
        self.num_nodes = 3
        self.shape_functions = []
        self.isoparametric = [1-s-t, s, t]
        self.isoparametric_nodes = [-1.0*s-1.0*t+1, 1.0*s , 1.0*t]
        self.x = self.isoparametric[0]*self.x1 + self.isoparametric[1]*self.x2+\
                self.isoparametric[2]*self.x3
        
        self.y = self.isoparametric[0]*self.y1 + self.isoparametric[1]*self.y2+\
                self.isoparametric[2]*self.y3

        self.elemental_stiffness = np.zeros([self.num_nodes,self.num_nodes])
        self.force_vector = np.zeros(self.num_nodes)
        
    def calc_shape_func(self):
        self.shape_functions = []
        for node_id in range(self.num_nodes):
            #N = np.array([0,0,0])
            N = Matrix([0,0,0])
            N[node_id] = 1
            # C = np.array([[1,self.x1,self.y1],\
            #               [1,self.x2,self.y2],\
            #               [1,self.x3,self.y3]])
            C = Matrix([[1,self.x1,self.y1],\
                        [1,self.x2,self.y2],\
                        [1,self.x3,self.y3]])
            #a = np.linalg.solve(C,N)
            a = C.solve(N)

            #function = round(a[0],2)+round(a[1],2)*x+round(a[2],2)*y
            function = a[0]+a[1]*x+a[2]*y
            self.shape_functions.append(function)
            #print(a)
            #print(N)
        return self.shape_functions

    def calc_jacobian_det(self):
        dx_ds = diff(self.x,s)
        dx_dt = diff(self.x,t)
        dy_ds = diff(self.y,s)
        dy_dt = diff(self.y,t)

        jacobian_matrix = [[dx_ds,dx_dt],\
                           [dy_ds,dy_dt]]

        jacobian_matrix = np.array(jacobian_matrix)
        
        #det_J = np.linalg.det(jacobian_matrix)
        det_J = jacobian_matrix[0,0]*jacobian_matrix[1,1] - jacobian_matrix[0,1]*jacobian_matrix[1,0]
        
        return abs(det_J)

    # def calc_partialN(self):
    #     dN_dx = []
    #     dN_dy = []
    #     for i in range(len(self.shape_functions)):
    #         dNdx = diff(self.shape_functions[i],x)
    #         dNdy = diff(self.shape_functions[i],y)
    #         dN_dx.append(dNdx)
    #         dN_dy.append(dNdy)

    #     N = Matrix([dN_dx,dN_dy])
    #     return N  
    def calc_partialN(self):
        dN_dx = []
        dN_dy = []
        self.shape_functions = []
        self.calc_shape_func()
        for i in range(len(self.shape_functions)):
            dNdx = diff(self.shape_functions[i],x)
            #print(dNdx)
            dNdy = diff(self.shape_functions[i],y)
            #print(dNdy)
            dN_dx.append(dNdx)
            dN_dy.append(dNdy)

        N = Matrix([dN_dx,dN_dy])
        return N 

    def calc_elemental_stiffness(self):
        B = self.calc_partialN()
        C = k
        #print(C)
        Bt = transpose(B)
        S = Bt*C*B
        K = zeros(S.rows,S.cols)
        #display(S)
        #print(len(S))
        #for i in S:
            #display(i)
        for i in range(S.rows):
            for j in range(S.cols):
                self.elemental_stiffness[i,j]= self.integrate_parametric(S[i,j])
                #K[i,j] = self.integrate_parametric(S[i,j])
        
        return self.elemental_stiffness

    def integrate_parametric(self,f):
        result = integrate(integrate(f*self.calc_jacobian_det(),(s,0,1-t)),(t,0,1))
        return result
        

    def element_flux(fx,fy,node1,node2): #the code currently accepts neumann boundary conditions with constant force term
        get_surface_normals(self.node_set)
        pass
        

#### class for meshing objects
#given a list of co-ordinates as nodes, this class creates a mesh for the given set of nodes and makes element objects for each
class Mesh:
    def __init__(self,node_set):   
        self.node_set = np.array(node_set)
        self.neumann = []
        self.dirichlet = []
        self.element_list = []
        self.ebc_nodes = [] # all nodes that have an essential boundary condition given to it
        self.num_nodes = np.shape(node_set)[0]
        self.global_stiffness_matrix = np.zeros([self.num_nodes,self.num_nodes])
        self.global_displacement = np.zeros(self.num_nodes)
        self.force_matrix = np.zeros(self.num_nodes)

    def element_objects(self): #creates triangular elements out of the node_set points
        tri = Delaunay(self.node_set)
        nodes = tri.simplices
        element_set = self.node_set[nodes]
        num_elements = np.shape(element_set)[0]
        self.element_list = []

        for i in range(num_elements):
            self.element_list.append(Trielement(element_set[i,0],element_set[i,1],element_set[i,2],nodes[i,0],nodes[i,1],nodes[i,2]))
        return self.element_list

    def get_element(self,i): # and auxillary method to help make a set of three points that form elements using delaunay triangulation
        #tri = Delaunay(self.node_set)
        #nodes = tri.simplices
        #element_set = self.node_set[nodes]
        #return element_set[i]
        element = self.element_list[i]
        x1 = np.array([element.x1 , element.y1])
        x2 = np.array([element.x2 , element.y2])
        x3 = np.array([element.x3 , element.y3])
        element_coordinates = np.array([x1,x2,x3])
        return element_coordinates
    
    # def calc_elemental_stiffness(self):
    #     pass
    #     for element in self.element_list:
    #         element.calc_elemental_stiffness

    def calc_global_stiffness(self):
        #assuming that you've already made element objects for your mesh
        for element in self.element_list:
            elemental_stiffness = element.calc_elemental_stiffness()
            print("Element No.", self.element_list.index(element))
            print("Nodes id used ", element.node_id)
            for i in range(3):
                for j in range(3):
                    #print()
                    self.global_stiffness_matrix[element.node_id[i],element.node_id[j]] += elemental_stiffness[i,j]

    def reduce_global_stiffness(self):
        for i, force in enumerate(self.force_matrix):
            if i in self.ebc_nodes:
                self.global_stiffness_matrix[self.ebc_nodes,:] = 0
                self.global_stiffness_matrix[:,self.ebc_nodes] = 0
                self.global_stiffness_matrix[self.ebc_nodes,self.ebc_nodes] = 1

    def plot_mesh(self, plot_element=True, plot_node=True, plot_temperature=False):  
        plt.figure(figsize=(8, 6))

        # Create a colormap for the temperature (displacement) values
        cmap = plt.cm.viridis

        # Get the min and max temperature for color normalization
        if plot_temperature:
            min_temp = np.min(self.global_displacement)
            max_temp = np.max(self.global_displacement)

        # Plot each element based on the coordinates in element_list
        for j, element in enumerate(self.element_list):
            x_vals = [element.x1, element.x2, element.x3, element.x1]
            y_vals = [element.y1, element.y2, element.y3, element.y1]

            # If plotting temperature, color the edges based on nodal temperature
            if plot_temperature:
                # Get the temperature values at each node
                temps = [self.global_displacement[element.id1], 
                         self.global_displacement[element.id2], 
                         self.global_displacement[element.id3]]

                for i in range(3):
                    # Define the two nodes of the edge (node i to node (i+1)%3)
                    x_edge = [self.node_set[element.node_id[i], 0], 
                              self.node_set[element.node_id[(i+1)%3], 0]]
                    y_edge = [self.node_set[element.node_id[i], 1], 
                              self.node_set[element.node_id[(i+1)%3], 1]]

                    # Define the temperatures at the two ends of the edge
                    temp_vals = [temps[i], temps[(i+1)%3]]

                    # Normalize the temperature values between 0 and 1 for the colormap
                    normed_temp1 = np.interp(temp_vals[0], [min_temp, max_temp], [0, 1])
                    normed_temp2 = np.interp(temp_vals[1], [min_temp, max_temp], [0, 1])

                    # Get the colors corresponding to the two temperature values
                    color1 = cmap(normed_temp1)  # Color for the first node
                    color2 = cmap(normed_temp2)  # Color for the second node

                    # Plot the first half of the edge (first node to mid-point)
                    midpoint_x = (x_edge[0] + x_edge[1]) / 2
                    midpoint_y = (y_edge[0] + y_edge[1]) / 2
                    plt.plot([x_edge[0], midpoint_x], [y_edge[0], midpoint_y], color=color1)

                    # Plot the second half of the edge (mid-point to second node)
                    plt.plot([midpoint_x, x_edge[1]], [midpoint_y, y_edge[1]], color=color2)

            else:
                plt.plot(x_vals, y_vals, color="black")

            # Calculate centroid of the triangle for labeling
            if plot_element:
                x_centroid = np.mean([element.x1, element.x2, element.x3])
                y_centroid = np.mean([element.y1, element.y2, element.y3])
                plt.text(x_centroid, y_centroid, f'{j}', color='red', ha='center', va='center')

        # Plot nodes and their labels
        if plot_node and not plot_temperature:
            plt.plot(self.node_set[:, 0], self.node_set[:, 1], 'o')
            for i, (x, y) in enumerate(self.node_set):
                plt.text(x, y, f'{i}', color='blue', ha='center', va='center')

        # Plot temperature at nodes if required
        if plot_temperature:
            sc = plt.scatter(self.node_set[:, 0], self.node_set[:, 1], c=self.global_displacement, cmap='viridis', s=100)
            plt.colorbar(sc, label='Temperature (or Displacement)')  # Add color bar for temperature
            for i, (x, y) in enumerate(self.node_set):
                plt.text(x, y, f'{i}', color='blue', ha='center', va='center')

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Finite Element Mesh with Nodal Temperature")
        plt.show()
        
    def neumann_to_element(self,i,j,fx,fy): #sets forces on elements with nodes i and j
        node1 = self.node_set[i]
        node2 = self.node_set[j]
        for element in element_list: #looks over the elements to see what elements share the nodes 1 and 2,
            pass


#auxillary functions

def get_normal(x1,x2): #a function to get normal vector given 2 points
    x1 = np.array(x1)
    x2 = np.array(x2)

    distance_vector = x2-x1

    normal_vector = ([-distance_vector[1],distance_vector[0]])

    norm = np.linalg.norm(normal_vector)

    if norm == 0:
        raise ValueError("The points cannot be the same")

    unit_normal = normal_vector/norm
    return unit_normal

def get_surface_normals(node_set):
    normals = np.zeros((node_set.shape[0], 2))  # Adjust size based on the number of nodes
    num_nodes = node_set.shape[0]
    first_normal = get_normal(node_set[0], node_set[1])
    
    for i in range(num_nodes):
        point1 = node_set[i]
        point2 = node_set[(i+1)%3]
        point3 = node_set[(i+2)%3]
        midpt = (point1+point2)/2
        direction_vector = point3-midpt
        
        #print(point1,point2,point3)
        #print(f"{midpt,direction_vector}+ asda")
        current_normal = get_normal(node_set[i], node_set[(i + 1) % num_nodes])
        #print(current_normal)
        if np.dot(direction_vector, current_normal) > 0:
            current_normal = -current_normal
        normals[i] = current_normal
    return normals

def plot_element_with_normals(node_set): #auxillary function to visualize the element and 
    normals = get_surface_normals(node_set)
    num_nodes = node_set.shape[0]
    midpoints = [(node_set[i] + node_set[(i + 1) % num_nodes]) / 2 for i in range(num_nodes)]

    #plot the triangle
    plt.plot(*zip(*np.append(node_set, [node_set[0]], axis=0)), 'k-')  # Close the loop for the triangle
    plt.plot(node_set[:, 0], node_set[:, 1], 'bo')  # Plot the vertices
    
    # Plot normals as vectors from the midpoints
    for i, midpoint in enumerate(midpoints):
        normal = normals[i]
        plt.quiver(midpoint[0], midpoint[1], normal[0], normal[1], angles='xy', scale_units='xy', scale=0.5, color='red')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Triangle with Surface Normals")
    plt.axis("equal")
    plt.grid(True)
    plt.show()