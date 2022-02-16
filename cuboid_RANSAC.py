# functions file
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import spatial
import itertools
from plyfile import PlyData, PlyElement
import numpy.matlib

DebugLevel = None

def boundTest(X, Y, Z, pp, corners):

  # corner: four 3d corners of a rectangle 4x3
  d1 = corners[0,:] - corners[1,:]
  d2 = corners[2,:] - corners[1,:]

  # po = [X,Y,Z] for plotting
  d1 = d1/np.linalg.norm(d1)
  d2 = d2/np.linalg.norm(d2)
  
  A = np.sum(np.multiply(pp, np.matlib.repmat(d1,np.size(X),1)),axis=1)
  B = np.sum(np.multiply(pp, np.matlib.repmat(d2,np.size(X),1)),axis=1)

  l1 = min(np.sum(d1.conj()*corners[0,:]),np.sum(d1.conj()*corners[1,:]))
  l2 = max(np.sum(d1.conj()*corners[0,:]),np.sum(d1.conj()*corners[1,:]))
  l3 = min(np.sum(d2.conj()*corners[2,:]),np.sum(d2.conj()*corners[1,:]))
  l4 = max(np.sum(d2.conj()*corners[2,:]),np.sum(d2.conj()*corners[1,:]))

  result = [(A<l2) & (A>l1) & (B<l4) & (B>l3)]

  return result

def ComplemenatryVec(vec, no):
    
    a = np.zeros((no))
    a[[vec]] = 1
    a_temp = (1-a).astype('int')
    temp2 = np.nonzero(a_temp)
    return temp2

def corner2para(corners):

    i = corners[3,:]-corners[1,:]
    j = corners[0,:]-corners[1,:]
    k = corners[5,:]-corners[1,:]

    centre = np.mean(corners, 0)

    l = math.sqrt(sum((i)**2))
    w = math.sqrt(sum((j)**2))
    h = math.sqrt(sum((k)**2))

    scale = np.array([l, w, h])

    i = i/np.linalg.norm(i)
    j = j/np.linalg.norm(j)
    k = k/np.linalg.norm(k)
    if math.sqrt(sum((np.cross(i,j)+k)**2)) < 0.2: # ponder over this number 0.2 (2 or e-16) 
      k = -k

    R = np.array([i, j, k])
    # orientation  = np.rad2deg(rpy(R)')
    orientation = np.transpose(rpy(R))*180/math.pi

    para = np.array([centre, scale, orientation])
    para = para.flatten()

    return para

def CuboidRANSAC(points3D):
  #CuboidRANSAC - Computes the best fitting cuboid on the 3D points using RANSAC. 
  # If 90 percent points are in the consensus set or 300 iterations are
  # reached, the code terminates and the current best is returned.
  #
  # Inputs: 
  #    points3D - nx3 Matric containing the 3D points placed in rows.
  #               points3D = [X, Y, Z]
  #
  # Outputs:
  #    model - 8x3 Matric containing the corners of the best-fit cuboid
  #    CuboidParameters - 9x1 Matric containing the parameters of the best-fit.
  #                Namely, centre, scale and orientation.
  #                [centre_x centre_y centre_z length width height roll pitch yaw]
  #    inlierIndicies - Linear indices of inlier points within the input point cloud,
  #                returned as a column vector.
  #    outlierIndices - Linear indices of outlier points within the input point cloud,
  #                returned as a column vector.
  #
  # Author: Usama Mehmood, Graduate Student, Stony Brook University, NY, US
  # email address: umehmood@cs.stonybrook.edu 
  # Website: https://usamamehmood.weebly.com
  # Novemnber 2014; Last revision: 23-Nov-2017
  #------------- BEGIN CODE --------------

  # Needs to be numpy array
  X = points3D[:, 0]
  Y = points3D[:, 1]
  Z = points3D[:, 2]

  dia = math.sqrt((max(X) - min(X))**2 + (max(Y) - min(Y))**2 + (max(Z) - min(Z))**2)
  num = dia/300 # Tolerance for consensus set.

  scoreLimit = math.ceil(0.9 * np.size(X))
  maxIterations = 250

  score = 0
  i =0
  Finalscore = 0
  model = 0
  total = 0
  n_final = 0
  Finalcset = []

  while(score < scoreLimit and i < maxIterations):
      [output, returnValue, _, total_cuboids]  = minSet( X,Y,Z )
      total = total + total_cuboids
      if output == 1:
          s = np.shape(returnValue)
          j = s[0]
          for k in range(0, j, 1):
              answer = planePlot(returnValue[k,:,:] ,0)
              para = corner2para(answer)
              [ _, score , cset, n_points] = RansacScore(num,X,Y,Z,para )
              if score > Finalscore and n_points > 100: #min number of points 100
                  Finalscore = score
                  model = answer
                  FinalPara = para
                  Finalcset = cset
                  n_final = n_points
  #                 Finaltemp = temp; %minimal set of points
  #                 FinalsumOfdist = sumOfdist;
  #                 Iteration = i;
      i = i+1
      if i%100 != 0:
          print('Iterations completed: ' + str(i))

  if Finalscore == 0:
    model = answer
    FinalPara = para
    Finalcset = cset

  print('Total Iterations: ' + str(i))
  print(Finalscore)
  print(n_final)
  Indices = np.arange(np.size(X))
  inlierIndices = np.transpose(np.extract(Finalcset != 0, Indices))
  outlierIndices = np.transpose(np.extract(Finalcset == 0, Indices))
  CuboidParameters = FinalPara

  return model, CuboidParameters, inlierIndices, outlierIndices, Finalscore
  
def DisplayModel(Finalscore, model, points3D, inlierIndices, outlierIndices):
  #CuboidRANSAC - Computes the best fitting cuboid on the 3D points using RANSAC. 
  # If 90 percent points are in the consensus set or 300 iterations are
  # reached, the code terminates and the current best is returned.
  #
  # Inputs: 
  #    model - 8x3 Matric containing the corners of the best-fit cuboid
  #    Finalscore - Number of points in the consensus set.
  #
  # Outputs:
  #    None
  #
  # Author: Usama Mehmood, Graduate Student, Stony Brook University, NY, US
  # email address: umehmood@cs.stonybrook.edu 
  # Website: https://usamamehmood.weebly.com
  # Novemnber 2014; Last revision: 23-Nov-2017
  #------------- BEGIN CODE --------------

  X = points3D[:,0]
  Y = points3D[:,1]
  Z = points3D[:,2]

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  array = np.array([[4,5,7,6], [2,3,7,6], [0,1,3,2], [0,1,5,4], [1,3,7,5], [0,2,6,4]])
  # fig.title(Finalscore)

  for j in range(0, 6, 1):
      u1 = array[j,:]
      B = model[u1,:]
      verts = [list(zip(B[:,0],B[:,1],B[:,2]))]
      ax.add_collection3d(Poly3DCollection(verts,color='green', alpha=0.5))


  ax.scatter(X[inlierIndices], Y[inlierIndices], Z[inlierIndices], color='blue', alpha =0.2)
  ax.scatter(X[outlierIndices], Y[outlierIndices], Z[outlierIndices], color='red', alpha =0.2)

  numberss = '12345678'
  #for i in range(0, 8, 1):
  #  fig.text(model[i,0],model[i,1],model[i,2],numberss[i])
  
  plt.show()
  return

def dist2plane(po, para):

  s = np.shape(po)
  rows = s[0]
  n = para[0:3]
  d = -para[3]

  dz = abs(np.sum(np.transpose(po).conj()*np.matlib.repmat(np.vstack(n),1,rows), axis=0) - d)
  dz = dz.reshape((-1,1))
  map = dz<=d
  map = map + (-1)*np.where(map == 0, 1, 0)
  pp = po + np.multiply(np.matlib.repmat(np.multiply(dz, map),1,3), np.matlib.repmat(n,rows,1))

  dz = np.transpose(dz)

  return pp, dz

def isequalf(a, b, thresh=2.2204e-16*100):
  #ISEQUALF  Returns true if the two quantities are equal within a threshold
  #
  #	t = ISEQUALF(A, B)
  #	[t reason] = ISEQUALF(A, B, THRESH)
  #
  # This function is useful for floating point values, where there may be
  # a small difference.
  #
  # See also: ISEQUAL.
  # $Id: isequalf.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
  # Copyright (C) 2005, by Brad Kratochvil
  
  #Disabled as there are no 'sym' type in Python
  #if isinstance(a, 'sym') or isinstance(b, 'sym'):
  #  if a == b:
  #    b = True
  #  else:
  #    b = False
  #else:
  m = np.amax(abs(a-b))
  if m > thresh:
    b = False
  else:
    b = True

  return b

def isrot(r):
  #ISROT  returns true if the matrix is a rotation matrix
  #
  #	T = ISROT(R)
  #	[T REASON] = ISTROT(R)
  #
  # if REASON is supplied as an output, a text based message as to why the
  # test failed will be returned.
  #
  # See also: ISTWIST, ISSKEW, ISHOM.#
  # $Id: isrot.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
  # Copyright (C) 2005, by Brad Kratochvil

  global DebugLevel

  out = True
  varargout = {'rot'}
  
  if (3, 3) != np.shape(r):
    out = False
    varargout = {'matrix must be 3x3'}
    return out, varargout
  
  if not DebugLevel or DebugLevel > 1:
    
    if not isequalf(np.identity(3), np.matmul(np.transpose(r), r), 1e-11):
      out = False
      varargout = {'transpose(r) * r = I violated'}
      return out, varargout

    if not isequalf(1, np.linalg.det(r)):
      out = False
      varargout = {'det(r) == 1 violated'}
      return out, varargout

  return out, varargout

def ComplemenatryVec(vec, no):
    
    a = np.zeros((no))
    a[[vec]] = 1
    a_temp = (1-a).astype('int')
    temp2 = np.nonzero(a_temp)
    return temp2

def dist2plane(po, para):

  s = np.shape(po)
  rows = s[0]
  n = para[0:3]
  d = -para[3]

  dz = abs(np.sum(np.transpose(po).conj()*np.matlib.repmat(np.vstack(n),1,rows), axis=0) - d)
  dz = dz.reshape((-1,1))
  map = dz<=d
  map = map + (-1)*np.where(map == 0, 1, 0)
  pp = po + np.multiply(np.matlib.repmat(np.multiply(dz, map),1,3), np.matlib.repmat(n,rows,1))

  dz = np.transpose(dz)

  return pp, dz

def minSet(X,Y,Z):
  # output is a flag for a valid minset, returnValue contains 6 planes (6x4xi)
  # (cuboid) fit in the minimal set, temp contains min set.
  # total_cuboids = total cuboids fitted in the min. set.
  returnValue = np.zeros((1,4,6))
  tempReturn = np.zeros((1,4,6))
  output = 0
  answerNo = 0
  no = 9
  indices = np.arange(no)
  finalResult = np.zeros((4,6))
  pnts = np.random.choice(np.size(X),no)
  temp = np.empty((np.size(pnts), 3))
  j = 0
  for i in pnts:
    temp[j] = np.array([[X[i], Y[i], Z[i]]])
    j = j + 1
  O = np.ones((no, 1))
  # Convex hull of 9 points
  K = spatial.ConvexHull(points = temp)
  num = np.shape(K.simplices)[0]

  distMax1 = np.zeros((1,num))
  planes  = np.zeros((4,num))
  rslt1 = np.zeros((num, len(list(itertools.combinations(range(no-3), 2)))))

  # Finds point that is max d away from every plane.
  for i in range(0, num, 1):
    temp1 = temp[K.simplices[i,0],:]
    temp2 = temp[K.simplices[i,1],:]
    temp3 = temp[K.simplices[i,2],:]
    n = np.cross((temp2-temp1),(temp3-temp1)) # normal Plane1
    if np.linalg.norm(n) == 0:
      continue

    n = n/np.linalg.norm(n)
    d = np.sum(temp1.conj()*n, axis=0)
    d = round(d*(10**4))/(10**4)
    planes[:,i] = np.append(n, -d)
    C = np.append(n, -d)
    [_, dz] = dist2plane(temp,C)
    indices = indices.reshape((1, 9))
    A1 = np.extract(dz-np.amax(dz)==0, indices)
    distMax1[:, i] = A1[0]
    n2 = n
    d2 = np.sum(n2.conj()*temp[int(distMax1[:,i]),:], axis=0)

    vec = K.simplices[i,:]

    Rem = ComplemenatryVec(vec, no)
    CmbRem = [x for x in itertools.combinations(Rem[0],2)] # All size 2 combinations of remaining 6 points
    CmbRem = np.array(CmbRem)

    # To find a plane2(Contains 2 points) perpendicular to plane1
    for k in range(0, int(np.size(CmbRem)/2), 1):
      vec1 = CmbRem[k,:]
      n1 = np.cross(n, (temp[vec1[0],:]- temp[vec1[1],:] ) ) # normal Plane2
      if np.linalg.norm(n1) == 0:
        continue
      n1 = n1/np.linalg.norm(n1)
      d1 = np.sum(n1.conj()*temp[vec1[0],:], axis=0)
      d1 = round(d1*10**4)/10**4
      t = O*n1
      t1 = np.sum(np.multiply(temp, t), axis=1)
      t1 = (t1*10**4).round()/10**4
      a = t1<d1
      c = sum(a)
      rslt1[i,k] = c

      if c == 0 or c==(no-2): # for legitimate planes go on to fit the cuboid
        finalResult[:,0] = planes[:,i] #1
        finalResult[:,1] = np.append(n1, -d1) # 2
        finalResult[:,2] = np.append(n2, -d2) # 3

        # C = [-n1(1)/n1(3) -n1(2)/n1(3) d1/n1(3)];
        C = np.append(n1, -d1)
        [projections, dz] = dist2plane(temp,C)
        A2 = np.extract(dz-np.amax(dz)==0, indices)
        d3 = np.sum(n1.conj()*temp[A2[0] ,:], axis=0)
        finalResult[:,3] = np.append(n1, -d3) #4

        n4 = np.cross(n,n1)
        n4 = n4/np.linalg.norm(n4)
        t = O*n4
        qwerty = sum(np.multiply(projections, t),2)
        Amin = np.extract(qwerty - np.amin(qwerty)==0, indices)
        minNum = Amin[0]
        Amax = np.extract(qwerty - np.amax(qwerty)==0, indices)
        maxNum = Amax[0]
        d4 = np.sum(n4.conj()*temp[minNum,:], axis=0)
        d5 = np.sum(n4.conj()*temp[maxNum,:], axis=0)
        finalResult[:,4] = np.append(n4, -d4) #5
        finalResult[:,5] = np.append(n4, -d5) #6
        # To see how many points are used in the cuboid
        vector1 = np.append(K.simplices[i,:], CmbRem[k,:])
        vector1 = np.append(vector1, distMax1[:,i])
        vector1 = np.append(vector1, [minNum, maxNum, A2[0]])
        nps = np.size(ComplemenatryVec(vector1.astype(int),no))

        if nps == 0:
          # scatter3(temp(:,1),temp(:,2),temp(:,3),'k','o');
          answerNo = answerNo + 1
          # hold on
          output = 1
          # PlotPlane( finalResult,0); % Plot final result.
          tempReturn[0,:,:] = finalResult
          if answerNo == 1:
            returnValue[answerNo-1,:,:] = tempReturn[0, :,:]
          else:
            returnValue = np.append(returnValue, tempReturn, axis=0)
                   
  total_cuboids = sum(sum((rslt1==0) + (rslt1==no-2)))

  return output, returnValue, temp, total_cuboids

def para2corner( para ):

  centre = para[0:3]
  scale = para[3:6]
  orientation= para[6:9]

  a = orientation[0]
  b = orientation[1]
  c = orientation[2]

  a_rad = np.deg2rad(a)
  b_rad = np.deg2rad(b)
  c_rad = np.deg2rad(c)

  Rx = np.array([[1, 0, 0], [0, np.cos(a_rad), -np.sin(a_rad)], [0, np.sin(a_rad), np.cos(a_rad)]])
  Ry = np.array([[np.cos(b_rad), 0, np.sin(b_rad)], [0, 1, 0], [-np.sin(b_rad), 0, np.cos(b_rad)]])
  Rz = np.array([[np.cos(c_rad), -np.sin(c_rad), 0], [np.sin(c_rad), np.cos(c_rad), 0], [0, 0, 1]])

  R = np.matmul(np.matmul(Rz, Ry), Rx)

  map = np.array([[-1,1,-1],[-1,-1,-1],[1,1,-1],[1,-1,-1],[-1,1,1],[-1,-1,1],[1,1,1],[1,-1,1]])
  corners = np.matlib.repmat(centre,8,1)+ np.matmul(map, ( np.multiply(R, ( 0.5*np.matlib.repmat(np.vstack(scale),1,3)))))

  return corners

def planePlot( b , flag ):
  # Solves for points of intersection of 6 planes.
  # if flag = 1 then plot the six planes , if zero then do not plot.
  answer = np.zeros((20,3))
  normals = b[0:3, :]
  normals = np.transpose(normals)
  d = b[3,:]
  Cmb = np.array(list(itertools.combinations(range(6), 3)))
  for i in range(0, 19, 1):
      A = normals[Cmb[i,:],:]
      c = -d[Cmb[i,:]]
      if abs(np.linalg.det(A))>0.005:
        answer[i,:] = np.linalg.lstsq(A, np.transpose(c))[0]

  v1 = np.nonzero(np.sum(answer, 1) != 0)
  # disp(v1)
  answer = answer[v1,:]
  # Don't know why, but it adds the dimension 1 to the depth of the answer matrix
  if len(np.shape(answer)) == 3:
    row_n = np.shape(answer)[1]
    col_n = np.shape(answer)[2]
    answer = np.reshape(answer, (row_n, col_n))

  # Disabled for now. Need to find equivalent for fill3 in python.
  #if flag == 1:
  #    Cmb = Cmb[v1,:]
  #    f = [1, 2, 4, 3]
  #    cl = ['b','b','b','b','b','b']
  #    for j in range(0, 6, 1):
  #        u1 = find(np.multiply(np.transpose(np.arange(0, np.size(Cmb)/3)), sum(Cmb==j,2)))
  #        u1 = u1(f)
  #        B = answer[u1,:]
  #        fill3(B[:,1], B(:,2),B(:,3),cl(j))
  #        alpha 0.5
  #        hold on
  #        axis equal

  return answer

def RansacScore(num,X,Y,Z,para):

  answer = np.array(para2corner(para))
  array = np.array([[4,5,7,6],[2,3,7,6],[0,1,3,2],[0,1,5,4],[1,3,7,5],[0,2,6,4]])
  po = np.transpose(np.array([X,Y,Z]))
  score = 0
  indices = np.arange(0,np.size(X))
  cset = np.zeros((np.size(X),1))
  sumOfdist = 0

  for i in range(0, 6, 1):
      corners = answer[array[i,:],:]
      d1 = corners[0,:]-corners[1,:]
      d2 = corners[2,:]-corners[1,:]
      n = np.cross(d1,d2) # normal
      n = n/np.linalg.norm(n)
      d = np.sum(corners[1,:].conj()*n, axis=0)
      d=round(d*10**4)/10**4
      C = np.append(n, -d)

      [pp, dz] = dist2plane(po,C)
      sumOfdist = sumOfdist + sum(dz[dz<3*num])
      points = sum(1 for d in dz[0] if d<3*num)

      result = boundTest(X,Y,Z,pp,corners)

      for i in range(len(cset)):
        cset[i] = cset[i] or (result[0][i] and dz[0][i]<num)
        if (result[0][i] and dz[0][i]<num):
          score = score + math.exp(-dz[0][i] * dz[0][i] / (2/ 9 * num * num))
        
      dz = dz[0][indices[result]]
  # score = sum(cset)
  
  # cset = indices(cset);
  #     score = sum(cset);

  return sumOfdist, score , cset, points


def rpy(R):

    #RPY  returns the X-Y-Z fixed angles of rotation matrix R
    #
    #	[ROLL PITCH YAW] = RPY(R)
    #
    # R is a rotation matrix. xyz is of the form [roll pitch yaw]'
    #
    # See also: .

    # $Id: rpy.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
    # Copyright (C) 2005, by Brad Kratochvil

    if not isrot(R)[0]:
      raise Exception('R is not a rotation matrix')
    
    beta = math.atan2(-R[2,0], math.sqrt(R[0,0]**2 + R[1,0]**2))

    if isequalf(beta, math.pi/2):
      alpha = 0
      gamma = math.atan2(R[0,1], R[1,1])
    elif isequalf(beta, -math.pi/2):
      alpha = 0
      gamma = -math.atan2(R[0,1], R[1,1])
    else:
      alpha = math.atan2(R[1,0]/math.cos(beta), R[0,0]/math.cos(beta))
      gamma = math.atan2(R[2,1]/math.cos(beta), R[2,2]/math.cos(beta))

    roll = gamma
    pitch = beta
    yaw = alpha

    xyz = np.transpose([roll, pitch, yaw])

    return xyz

def evaluate_cuboid(ply_path):
  # dir = input("Insert point cloud location: ")
  pc = PlyData.read(ply_path)
  X = pc['vertex']['x']
  Y = pc['vertex']['y']
  Z = pc['vertex']['z']
  points3D = np.transpose(np.array([X, Y, Z]))
  print(np.shape(points3D))
  [model, CuboidParameters, inlierIndices, outlierIndices, Finalscore] = CuboidRANSAC(points3D)
  print("Number of inlier points: ", np.shape(inlierIndices))
  print("Number of outlier points: ", np.shape(outlierIndices))
  # print("Total number of points: ", np.shape(points3D))

  DisplayModel(Finalscore, model, points3D, inlierIndices, outlierIndices)

  update_points3D = np.transpose(np.array([X[outlierIndices], Y[outlierIndices], Z[outlierIndices]]))
  return CuboidParameters

  [model_2, CuboidParameters_2, inlierIndices_2, outlierIndices_2, Finalscore_2] = CuboidRANSAC(update_points3D)

  DisplayModel(Finalscore_2, model_2, update_points3D, inlierIndices_2, outlierIndices_2)


path = input("Insert path of the ply file: ")
evaluate_cuboid(path)