import numpy as np
import matplotlib.pyplot as plot
import random as rand
'''
width (float): width of the rectangle boundary
height (float): height of the rectangle boundary
sites (list[points]): a list of sites (points): (x,y) from given data
Assume the four vertices of the rectangle are (0,0), (width,0), (width,height), (0,height)
'''
def generate_diagrams(width, height, sites):
    E = []  # list of edges
    C = []  # list of cells
    # compute perpendicular bisector of a line within the box
    def perp_bisec(p1, p2):
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        # slope of the perp bisector line
        if y1 - y2 == 0:
            edge = [((x1 + x2)/2, 0), ((x1 + x2)/2, height)]
        elif x1 - x2 == 0:
            edge = [(0, (y1 + y2)/2), (width, (y1 + y2)/2)]
        else:
            slope = (x2 - x1)/(y1 - y2) 
            # mid-point of p1, p2
            mid_point = ((x1 + x2)/2, (y1 + y2)/2)  
            h1 = mid_point[1] - slope*mid_point[0] 
            h2 = mid_point[1] + slope*(width - mid_point[0]) 
            w1 = mid_point[0] - mid_point[1]/slope
            w2 = mid_point[0] + (height - mid_point[1])/slope
            if 0 <= w1 < width and 0 <= h2 < height:
                edge = [(width, h2), (w1, 0)]
            if 0 <= w1 < width and 0 < w2 <= width:  
                edge = [(w1, 0), (w2, height)]
            if 0 <= w1 < width and 0 < h1 <= height:
                edge = [(0, h1), (w1, 0)]
            if 0 <= h2 < height and 0 < w2 <= width:
                edge = [(width, h2), (w2, height)]
            if 0 <= h2 < height and 0 < h1 <= height:
                edge = [(0, h1), (width, h2)]
            if 0 < w2 <= width and 0 < h1 <= height:
                edge = [(0, h1), (w2, height)]
        return edge
    # initialize the first cell using first two sites
    edge = perp_bisec(sites[0], sites[1])
    E.append(edge)
    cell0 = [sites[0], edge]
    cell1 = [sites[1], edge]
    [p1, p2] = edge
    boundary_set = [[(0, 0), (width, 0)], [(width, 0), (width, height)], 
                    [(width, height), (0, height)], [(0, height), (0, 0)]]
    for k in range(len(boundary_set)):
        par = k%2   # 0 or 1
        neg_par = (par + 1)%2   # 1 or 0
        e = boundary_set[k]
        if abs(p1[neg_par] - e[0][neg_par]) == 0 and \
            (e[0][par] - p1[par])*(e[1][par] - p1[par]) < 0:    # p1 on this boundary edge e
            if abs(sites[0][par] - e[0][par]) < abs(sites[1][par] - e[0][par]):
                cell0.append([e[0], p1])
                cell1.append([e[1], p1])
            else:
                cell0.append([e[1], p1])
                cell1.append([e[0], p1])
        elif abs(p2[neg_par] - e[0][neg_par]) == 0 and \
            (e[0][par] - p2[par])*(e[1][par] - p2[par]) < 0:  # p2 on this boundary edge e
            if abs(sites[0][par] - e[0][par]) < abs(sites[1][par] - e[0][par]):
                cell0.append([e[0], p2])
                cell1.append([e[1], p2])
            else:
                cell0.append([e[1], p2])
                cell1.append([e[0], p2])
        else:   # neither p1 or p2 on this boundary edge e
            if abs(sites[0][neg_par] - e[0][neg_par]) < abs(sites[1][neg_par] - e[0][neg_par]):
                cell0.append(e)
            else:
                cell1.append(e)
    C.extend([cell0, cell1])
    
    # Start the process for each site
    for k in range(2, len(sites)):
        site = sites[k]
        cell = [site]
        
        # Trim an edge to near site of c in C or delete if necessary
        def update_edge(c_site, site, edge):
            p1, p2 = edge[0], edge[1]
            dis_to_c_1 = (p1[0] - c_site[0])**2 + (p1[1] - c_site[1])**2
            dis_to_site_1 = (p1[0] - site[0])**2 + (p1[1] - site[1])**2
            dis_to_c_2 = (p2[0] - c_site[0])**2 + (p2[1] - c_site[1])**2
            dis_to_site_2 = (p2[0] - site[0])**2 + (p2[1] - site[1])**2
            if dis_to_c_1 >= dis_to_site_1 and dis_to_c_2 >= dis_to_site_2:
                return 'delete' # this edge will be deleted
            elif dis_to_c_1 >= dis_to_site_1 and dis_to_c_2 < dis_to_site_2:
                return [intersect(edge, perp_bisec(c_site, site)), p2]
            elif dis_to_c_1 < dis_to_site_1 and dis_to_c_2 >= dis_to_site_2:
                return [intersect(edge, perp_bisec(c_site, site)), p1]
            else:
                return 'do nothing' # do nothing
        for c in C: # for each existing cell
            c_site = c[0]
            
            critical_points = []
            for j in range(1, len(c)): # for each edge in c
                #print("site", site, "c-site", c_site)
                #print('Acting on edge', c[j])
                act = update_edge(c_site, site, c[j])
                #print(act)
                if act == 'delete':
                    to_delete = c[j] # delete from the edge set first
                    E[:] = [e for e in E if e != to_delete]
                    c[j] = None # perform deletion on c later 
                elif act == 'do nothing':
                    pass
                else:   # update the edge in c, E
                    critical_points.append(act[0])
                    E[:] = [act if e == c[j] else e for e in E]
                    c[j] = act

            if len(critical_points) == 2 and \
                critical_points[0] != critical_points[1]: # in case two critical points overlap
                new_edge = critical_points
                c.append(new_edge)  # add to the existing cell in the iteration
                cell.append(new_edge)   # add to the contructing cell
                E.append(new_edge)  # add to the edge set
            c[:] = [e for e in c if e != None]
            #print(c)
        complete(cell, width, height)
        C.append(cell)
    return E, C




# Test if two line segements intersect
# return the intersection if so, otherwise return None
def intersect(edge1, edge2):
    (x1, y1), (x2, y2) = edge1[0], edge1[1]
    (x3, y3), (x4, y4) = edge2[0], edge2[1]
    A = [[x2 - x1, x3 - x4], [y2 - y1, y3 - y4]]
    b = [x3 - x1, y3 - y1]
    [t, s] = np.linalg.solve(A, b)  # if |t|, |s| <= 1, two edges intersect
    if abs(t) > 1 or abs(s) > 1:
        return None
    else:
        x = x1 + t*(x2 - x1)
        y = y1 + t*(y2 - y1)
        return (x, y)


# Complete the boundary edges of a cell
def complete(cell, width, height):
    s = cell[0] # site of the cell
    points = []
    edges = []
    for k in range(1, len(cell)):
        edge = cell[k]
        points += [p for p in edge if (p[0] == 0 and 0 < p[1] <= height) or \
                                    (p[0] == width and 0 <= p[1] < height) or \
                                    (p[1] == height and 0 < p[0] <= width) or \
                                    (p[1] == 0 and 0 <= p[0] < width)]
    boundary_set = [[(0, 0), (width, 0)], [(width, 0), (width, height)], 
                    [(width, height), (0, height)], [(0, height), (0, 0)]]
    points_set = set(points)
    if len(points) == 2: # this cell is at the boundary and needs to be completed
        edges += [e for e in cell[1:] if points[0] == e[0] or points[0] == e[1]\
                                    or points[1] == e[0] or points[1] == e[1]]
        def left_or_right(u, v): # test u is on the left or right of v
            x1, y1, x2, y2 = u[0], u[1], v[0], v[1]
            if -x1*y2 + x2*y1 > 0:
                return 'left'
            return 'right'
        p1, p2 = points_set.pop(), points_set.pop()
        if p2[0] - p1[0] == 0 or p2[1] - p1[1] == 0:
            cell.append([p1, p2])
            return
        for k in range(len(boundary_set)):
            par = k%2   # 0 or 1
            neg_par = (par + 1)%2   # 1 or 0
            e = boundary_set[k]
            if abs(p1[neg_par] - e[0][neg_par]) == 0 and \
                (e[0][par] - p1[par])*(e[1][par] - p1[par]) < 0:    # p1 on this boundary edge e
                if edges[0][0] == p1:
                    p22 = edges[0][1]
                elif edges[0][1] == p1:
                    p22 = edges[0][0]
                elif edges[1][0] == p1:
                    p22 = edges[1][1]
                else:
                    p22 = edges[1][0]
                v1 = tuple(map(lambda i, j: i - j, p22, p1))
                v2 = tuple(map(lambda i, j: i - j, s, p1))
                if left_or_right(v2, v1) == 'left':
                    cell.append([e[0], p1])
                else:
                    cell.append([p1, e[1]])
            elif abs(p2[neg_par] - e[0][neg_par]) == 0 and \
                (e[0][par] - p2[par])*(e[1][par] - p2[par]) < 0:  # p2 on this boundary edge e
                if edges[0][0] == p2:
                    p11 = edges[0][1]
                elif edges[0][1] == p2:
                    p11 = edges[0][0]
                elif edges[1][0] == p2:
                    p11 = edges[1][1]
                else:
                    p11 = edges[1][0]
                v1 = tuple(map(lambda i, j: i - j, p11, p2))
                v2 = tuple(map(lambda i, j: i - j, s, p2))
                if left_or_right(v2, v1) == 'left':
                    cell.append([e[0], p2])
                else:
                    cell.append([p2, e[1]])
            else:   # neither p1 or p2 on this boundary edge e
                [q1, q2] = edges[0]
                if k == 0:
                    if q2[0] - q1[0] == 0: # vertical 
                        [q1, q2] = edges[1]
                    slope = (q2[1] - q1[1])/(q2[0] - q1[0])
                    intercept = q1[1] - slope*q1[0]
                    if s[1] < intercept + slope*s[0]:
                        cell.append(e)
                if k == 1:
                    if q2[1] - q1[1] == 0: # horizontal 
                        [q1, q2] = edges[1]
                    slope = (q2[0] - q1[0])/(q2[1] - q1[1])
                    intercept = q1[0] - slope*q1[1]
                    if s[0] > intercept + slope*s[1]:
                        cell.append(e)
                if k == 2:
                    if q2[0] - q1[0] == 0: # vertical 
                        [q1, q2] = edges[1]
                    slope = (q2[1] - q1[1])/(q2[0] - q1[0])
                    intercept = q1[1] - slope*q1[0]
                    if s[1] > intercept + slope*s[0]:
                        cell.append(e)
                if k == 3:
                    if q2[1] - q1[1] == 0: # horizontal 
                        [q1, q2] = edges[1]
                    slope = (q2[0] - q1[0])/(q2[1] - q1[1])
                    intercept = q1[0] - slope*q1[1]
                    if s[0] < intercept + slope*s[1]:
                        cell.append(e)

''' Sample data to test
E, C = generate_diagrams(4, 4, [(2, 1), (3, 2), (2, 4), (1, 2)])
print("outcome")
for each in C:
    print(each)
for each in E:
    print(each)
'''
size = 10   # you can change the number of random points
x, y = np.random.rand(size), np.random.rand(size)
width, height = 1, 1
sites = []
for j in range(size):
    sites.append((x[j], y[j]))
print('Random points are')
print(sites)
E, C = generate_diagrams(width, height, sites)
plot.scatter(x, y, color='red', s=6)
for e in E:
    plot.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]], color='black', linewidth=0.5)
plot.show()