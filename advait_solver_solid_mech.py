#disclosure: plot methods have been created with the help of gen ai
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

x = symbols('x')
y = symbols('y')
s = symbols('s')
t = symbols('t')
#k = Matrix([[1,0],[0,1]])

E = 200e9 #GPa
v = 0.3
thickness = 0.001 #m Thickness of the object is inherent to the governing differential equation.
u_nodes = [0,2,4]
v_nodes = [1,3,5]

#plane strain
# C = E/((1+v)*(1-2*v)) 
# c11 = 1-v 
# c12 = v  
# c21 = v 
# c22 = 1-v 
# c44 = (1-2*v)/2 

#plane stress
C = E/(1-v**2)
c11 = 1
c12 = v
c21 = v
c22 = 1
c44 = (1-v)/2


cij = C*Matrix([[c11,c12,0],\
                [c21,c22,0],\
                [0,0,c44]])

#### class for meshing objects
#given a list of co-ordinates as nodes, this class creates a mesh for the given set of nodes and makes element objects for each
class Mesh:
    def __init__(self,node_set):   
        self.node_set = np.array(node_set)
        self.neumann = []
        self.dirichlet = []
        self.element_list = []
        self.num_nodes = np.shape(node_set)[0]
        self.global_stiffness_matrix = np.zeros([2*self.num_nodes,2*self.num_nodes])
        self.global_displacement = np.zeros(2*self.num_nodes)
        self.force_matrix = np.zeros(2*self.num_nodes)
        self.u_nodes = 2*np.arange(self.num_nodes)
        self.v_nodes = self.u_nodes + 1
        self.ebc_nodes = [{'node': None, 'u_fixed' : None, 'v_fixed': None}]
        self.ebc_x = [] # nodes where ux is fixed
        self.ebc_y = [] # nodes where uy is fixed
        
    # def ebc_bool(self):
    #     for node in self.ebc_nodes:
    #         self.ebc_bool.append({'u_fixed' : False, 'v_fixed' : False})
    
    def element_objects(self): #creates triangular elements out of the node_set points
        tri = Delaunay(self.node_set)
        nodes = tri.simplices
        element_set = self.node_set[nodes]
        num_elements = np.shape(element_set)[0]
        self.element_list = []

        for i in range(num_elements):
            self.element_list.append(Trielement(element_set[i,0],element_set[i,1],element_set[i,2],nodes[i,0],nodes[i,1],nodes[i,2]))
        return self.element_list

    def calc_global_stiffness(self):
        global_u_nodes = self.u_nodes
        global_v_nodes = self.v_nodes

        for element in self.element_list:
            element.calc_elemental_stiffness()
            for i in range(3):
                for j in range(3):
                    node1 = element.node_id[i]
                    node2 = element.node_id[j]
            
                    self.global_stiffness_matrix[global_u_nodes[node1],global_u_nodes[node2]] += element.elemental_stiffness[u_nodes[i],u_nodes[j]]
                    self.global_stiffness_matrix[global_u_nodes[node1],global_u_nodes[node2]+1] += element.elemental_stiffness[u_nodes[i],u_nodes[j]+1]
                    self.global_stiffness_matrix[global_v_nodes[node1],global_v_nodes[node2]-1] += element.elemental_stiffness[v_nodes[i],v_nodes[j]-1]
                    self.global_stiffness_matrix[global_v_nodes[node1],global_v_nodes[node2]] += element.elemental_stiffness[v_nodes[i],v_nodes[j]]
                    
    def make_ebc_nodes(self):
        ebc_set = set(self.ebc_x + self.ebc_y)
        for i in ebc_set:
            if i in ebc_x and i not in ebc_y: 
                ebc_nodes.append({'node': i, 'u_fixed': True, 'v_fixed': False})
            elif i in ebc_y and i not in ebc_x: 
                ebc_nodes.append({'node': i, 'u_fixed': False, 'v_fixed': True})
            elif i in ebc_x and i in ebc_y: 
                ebc_nodes.append({'node': i, 'u_fixed': True, 'v_fixed': True})
        
        self.ebc_nodes.pop(0)
        
        return self.ebc_nodes

    def reduce_global_stiffness(self):
        for node in self.ebc_x:
            self.global_stiffness_matrix[self.u_nodes[node],:] = 0
            self.global_stiffness_matrix[:,self.u_nodes[node]] = 0
            self.global_stiffness_matrix[self.u_nodes[node],self.u_nodes[node]] = 1
                
        for node in self.ebc_y:
            self.global_stiffness_matrix[self.v_nodes[node],:] = 0
            self.global_stiffness_matrix[:,self.v_nodes[node]] = 0
            self.global_stiffness_matrix[self.v_nodes[node],self.v_nodes[node]] = 1
                
    
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

    def plot_mesh(self, plot_element=True, plot_node=True, plot_u=False, plot_v = False):  
        plt.figure(figsize=(8, 6))

        # Create a colormap for the temperature (displacement) values
        cmap = plt.cm.viridis

        # Get the min and max temperature for color normalization
        if plot_u:
            min_u = np.min(self.global_displacement)
            max_u = np.max(self.global_displacement)

        # Plot each element based on the coordinates in element_list
        for j, element in enumerate(self.element_list):
            x_vals = [element.x1, element.x2, element.x3, element.x1]
            y_vals = [element.y1, element.y2, element.y3, element.y1]

            # If plotting temperature, color the edges based on nodal temperature
            if plot_u:
                # Get the temperature values at each node
                u_s = [self.global_displacement[element.id1], 
                         self.global_displacement[element.id2], 
                         self.global_displacement[element.id3]]

                for i in range(3):
                    # Define the two nodes of the edge (node i to node (i+1)%3)
                    x_edge = [self.node_set[element.node_id[i], 0], 
                              self.node_set[element.node_id[(i+1)%3], 0]]
                    y_edge = [self.node_set[element.node_id[i], 1], 
                              self.node_set[element.node_id[(i+1)%3], 1]]

                    # Define the temperatures at the two ends of the edge
                    u_vals = [u_s[i], u_s[(i+1)%3]]

                    # Normalize the temperature values between 0 and 1 for the colormap
                    normed_u1 = np.interp(u_vals[0], [min_u, max_u], [0, 1])
                    normed_u2 = np.interp(u_vals[1], [min_u, max_u], [0, 1])

                    # Get the colors corresponding to the two temperature values
                    color1 = cmap(normed_u1)  # Color for the first node
                    color2 = cmap(normed_u2)  # Color for the second node

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
        if plot_node and not plot_u:
            plt.plot(self.node_set[:, 0], self.node_set[:, 1], 'o')
            for i, (x, y) in enumerate(self.node_set):
                plt.text(x, y, f'{i}', color='blue', ha='center', va='center')

        # Plot temperature at nodes if required
        if plot_u:
            sc = plt.scatter(self.node_set[:, 0], self.node_set[:, 1], c=self.global_displacement, cmap='viridis', s=100)
            plt.colorbar(sc, label='u displacement')  # Add color bar for temperature
            for i, (x, y) in enumerate(self.node_set):
                plt.text(x, y, f'{i}', color='blue', ha='center', va='center')

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Finite Element Mesh with Nodal X displacement")
        plt.show()

    # def plot_mesh(self,plot_element = True,plot_node = True):  # method to plot the mesh using the element_list attribute
    #     plt.figure(figsize=(8, 6))
        
    #     # Plot each element based on the coordinates in element_list
    #     for j, element in enumerate(self.element_list):
    #         x_vals = [element.x1, element.x2, element.x3, element.x1]
    #         y_vals = [element.y1, element.y2, element.y3, element.y1]
    #         plt.plot(x_vals, y_vals, color="black")

    #         # Calculate centroid of the triangle for labeling
    #         if plot_element:
    #             x_centroid = np.mean([element.x1, element.x2, element.x3])
    #             y_centroid = np.mean([element.y1, element.y2, element.y3])
    #             plt.text(x_centroid, y_centroid, f'{j}', color='red', ha='center', va='center')

    #     # Plot nodes and their labels
    #     if plot_node:
    #         plt.plot(self.node_set[:, 0], self.node_set[:, 1], 'o')
    #         for i, (x, y) in enumerate(self.node_set):
    #             plt.text(x, y, f'{i}', color='blue', ha='center', va='center')

    #     plt.xlabel("X Coordinate")
    #     plt.ylabel("Y Coordinate")
    #     plt.title("Finite Element Mesh with Node and Element Numbers")
    #     plt.show()
        
    def neumann_to_element(self,i,j,fx,fy): #sets forces on elements with nodes i and j
        node1 = self.node_set[i]
        node2 = self.node_set[j]
        for element in element_list: #looks over the elements to see what elements share the nodes 1 and 2,
            pass

class Trielement(Mesh):
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
        self.node_set = [x1,x2,x3]
        self.num_nodes = 3
        self.shape_functions = []
        self.isoparametric = [1-s-t, s, t]
        self.isoparametric_nodes = [-1.0*s-1.0*t+1, 1.0*s , 1.0*t]
        self.x = self.isoparametric[0]*self.x1 + self.isoparametric[1]*self.x2+\
                self.isoparametric[2]*self.x3
        
        self.y = self.isoparametric[0]*self.y1 + self.isoparametric[1]*self.y2+\
                self.isoparametric[2]*self.y3

        self.elemental_stiffness = np.zeros([2*self.num_nodes,2*self.num_nodes])
        self.force_vector = np.zeros(2*self.num_nodes)
        
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
            dNdy = diff(self.shape_functions[i],y)
            dN_dx.append(dNdx)
            dN_dy.append(dNdy)
        
        dN1dx = dN_dx[0]
        dN2dx = dN_dx[1]
        dN3dx = dN_dx[2]

        dN1dy = dN_dy[0]
        dN2dy = dN_dy[1]
        dN3dy = dN_dy[2]        
        
        N = Matrix([[dN1dx,0,dN2dx,0,dN3dx,0],\
                    [0,dN1dy,0,dN2dy,0,dN3dy],\
                    [dN1dy,dN1dx,dN2dy,dN2dx,dN3dy,dN3dx]])
        return N 

    def calc_elemental_stiffness(self):
        B = self.calc_partialN()
        C = cij
        Bt = transpose(B)
        S = Bt*C*B*thickness             #this thickness is multiplied to integrate the 2D equation in the domain
        K = zeros(S.rows,S.cols)
        #display(B)
        #display(C)
        #display(S)
        #print(len(S))
        #for i in S:
            #display(i)
        for i in range(S.rows):
            for j in range(S.cols):
                self.elemental_stiffness[i,j]= self.integrate_parametric(S[i,j])
                #K[i,j] = self.integrate_parametric(S[i,j])
        
        return self.elemental_stiffness#Bt*C*B
    
    # def elemental_stiffness(self):
    #     B = self.calc_partialN()
    #     C = k
    #     Bt = transpose(B)
    #     S = Bt*C*B
    #     K = zeros(S.rows,S.cols)
    #     display(S)
    #     #print(len(S))
    #     #for i in S:
    #         #display(i)
    #     for i in range(S.rows):
    #         for j in range(S.cols):
    #             K[i,j]= self.integrate_parametric(S[i,j])
    #     return K#Bt*C*B

    def integrate_parametric(self,f):
        result = integrate(integrate(f*self.calc_jacobian_det(),(s,0,1-t)),(t,0,1))
        return result

    def element_flux(fx,fy,node1,node2): #the code currently accepts neumann boundary conditions with constant force term
        get_surface_normals(self.node_set)
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