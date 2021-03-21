import numpy as np
from scipy.interpolate import griddata
from scipy.special import roots_legendre
from scipy.fft import fft
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cr3bp import Cr3bp, PeriodicOrbit

N1 = 20
N2 = 20
m = 5

# Retrieve seed periodic orbit data
element = 1
fileName1 = "northHaloL1.dat"
debugPlot = False

massParameter = 3.003480642487067e-06
distanceSunEarth = 149597870700 # in m
var = PeriodicOrbit("Sun","Earth", distanceSunEarth)
myDict = var.readDatabase(fileName1)
initialState, termPeriod  = list(myDict.values())[element]

# Propagate state
nullTerminationCondition = lambda t,x: t + 1E3
nullTerminationCondition.terminal = True
nullTerminationCondition.direction = -1
solData, stateTransition = var.runFullSimulationUntilTermination(initialState, 0.0, nullTerminationCondition, max_step=0.01, maxTime=2*termPeriod)
if debugPlot:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(solData.y[0], solData.y[1], solData.y[2],'blue')
    plt.show()

# Method to create initial torus
def getSeedTorus( N2, rho, K, eigenVector, solData, stateTransitionSet ):
    theta2Space = np.linspace(0, 1, N2, endpoint=False)
    # Create initial seed circle 
    seedCircle = np.array([K * (np.cos(2*np.pi*angle) * eigenVector.real - np.sin(2*np.pi*angle) * eigenVector.imag) for angle in theta2Space])
    
    # Propagate perturbations using state transition matrices
    torus = dict()
    timeSet = np.linspace(0, 1, len(stateTransitionSet))
    for num, stateTransition in enumerate(stateTransitionSet):
        time = timeSet[num] # normalized
        periodicState = solData.y[:,num]                        # unwinding term 
        torus.update({time: periodicState + np.array([np.real(np.exp(-1j * 2 * np.pi* rho * time)) * stateTransition @ state for state in seedCircle])})
    
    return torus

# Method to create collocation torus
def transformToCollocationTorus( torus, N1, N2, m, percentageKeep=1):
    angle = np.linspace(0, 1, N2, endpoint=False)
    step = int(1/percentageKeep)

    # Create grid tau/theta2 to interpolate
    timeList = list(torus.keys())
    time = np.concatenate([np.array(timeList)[::step], [timeList[-1]]])   # always add last element
    timeNorm = time / timeList[-1]
    
    gridInterpolate = [(a,b) for a in timeNorm for b in angle]
    # Create data to interpolate
    dataInterpolate = [state for aTime in time for state in torus[aTime]]

    # Create grid using Gauss-Legendre collocation points
    gaussLegrendreArray = createGaussLegendreCollocationArray(N1, m)
    xv, yv = np.meshgrid(gaussLegrendreArray, angle)
       
    # Create interpolator
    solInterpolator = griddata(gridInterpolate, dataInterpolate, (xv, yv), method='linear')
    # Return corrected torus
    collocationTorus = dict()
    for num, time in enumerate(gaussLegrendreArray):
        collocationTorus.update({time: [solInterpolator[angle,num] for angle in range(N2)]})

    return collocationTorus

# Creates an array with the collocation values between [0,1]
def createGaussLegendreCollocationArray( N, m ):
    collocationaArray = []

    for intervalValue in np.linspace(0, 1, N+1):
        collocationaArray.append(intervalValue)
        # Break the last iteration before filling
        if intervalValue == 1.0: break
        offset = intervalValue
        for root in roots_legendre(m)[0]:
            collocationaArray.append(offset + (root / 2.0 + 0.5) / N) # Tranform to the interval [0,1]

    return collocationaArray

# Method to draw a torus
def printTorus( torus ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for key in list(torus.keys()):
        circle = np.array(torus[key])
        if key == list(torus.keys())[0]:
            ax.plot(circle[:,0], circle[:,1], circle[:,2], 'red')
        elif key == list(torus.keys())[-1]:
            ax.plot(circle[:,0], circle[:,1], circle[:,2], 'blue')
        elif key in np.linspace(0, 1, N1 + 1):
            ax.plot(circle[:,0], circle[:,1], circle[:,2], 'green')
        else:
            ax.plot(circle[:,0], circle[:,1], circle[:,2], 'black')

# Get eigenvalues/vectors of monondormy matrix
monodromyMatrix = stateTransition[-1]
eigenValues, eigenVectors = np.linalg.eig(monodromyMatrix)
qpEigenvalue = eigenValues[2]
qpEigenvector = eigenVectors[:,2]
rho = np.angle(qpEigenvalue) / (2 * np.pi)
T = 2 * termPeriod

# Create collocation torus
firstTorus = getSeedTorus(N2, rho, 0.001, qpEigenvector, solData, stateTransition)
collocationTorus = transformToCollocationTorus(firstTorus, N1, N2, m)
if debugPlot:
    printTorus(collocationTorus)

# Find tangent solution
oldTanTorus = firstTorus
for numCircle, circleKey in enumerate(firstTorus.keys()):
    oldTanTorus[circleKey] -= solData.y[:, numCircle]
tanTorus = transformToCollocationTorus(oldTanTorus, N1, N2, m)

# Lagrange Polynomial tools
def tauHat( t, ti, tii ):
    return (t - ti) / (tii - ti)

def lagrangePol( time, timeSampleK, timeSegment):
    ti = timeSegment[0]
    tii = timeSegment[-1]
    sol = 1
    for timeJ in timeSegment[:-1]:
        if timeJ != timeSampleK:
            sol *= (tauHat(time, ti, tii) - tauHat(timeJ, ti, tii)) / (tauHat(timeSampleK,ti,tii) - tauHat(timeJ,ti,tii))

    return sol

def lagrangePolDer( time, timeSampleK, timeSegment):
    ti = timeSegment[0]
    tii = timeSegment[-1]

    sol1 = 0
    for timeJ1 in timeSegment[:-1]:
        if timeJ1 != timeSampleK:
            sol2 = 1
            for timeJ2 in timeSegment[:-1]:
                if timeJ2 != timeSampleK and timeJ2 != timeJ1:
                    sol2 *=  (tauHat(time, ti, tii) - tauHat(timeJ2, ti, tii)) / (tauHat(timeSampleK,ti,tii) - tauHat(timeJ2,ti,tii))
            sol2 *= 1. / (tauHat(timeSampleK,ti,tii) - tauHat(timeJ1,ti,tii))
            sol1 += sol2

    return sol1

# CR3BP dyanmical problem
def modelDynamicsCR3BP( t, state ):
    x, y, z, dx, dy, dz = state[0], state[1], state[2], state[3], state[4], state[5]
    u = massParameter
    r1 = np.sqrt((u + x)**2 + y**2 + z**2)
    r2 = np.sqrt((1 - u - x)**2 + y**2 + z**2)

    ddx = x - (u + x) * (1 - u) / r1**3 + (1 - u - x) * u / r2**3 + 2 * dy
    ddy = y       - y * (1 - u) / r1**3           - y * u / r2**3 - 2 * dx
    ddz =         - z * (1 - u) / r1**3           - z * u / r2**3

    return np.array([dx, dy, dz, ddx, ddy, ddz])

# Method to compute matrix to compute dth2
def th2DerMatrix( N ):
    vect = [1j * k for k in np.concatenate([np.arange(0,N//2), [0], np.arange(-N//2+1,0)])]
    dft = fft(np.eye(N))
    dth2DerMatrix = np.linalg.inv(dft) @ (( np.eye(N).T* vect).T @ dft)

    return expandMatrix(dth2DerMatrix.real, N)

# Method to compute the gradient of the Jacobi energy
def jacobiGradient( state ):
        u = massParameter
        x, y, z, dx, dy, dz = state[0], state[1], state[2], state[3], state[4], state[5]
        r1 = np.sqrt((u + x)**2 + y**2 + z**2)
        r2 = np.sqrt((1 - u - x)**2 + y**2 + z**2)

        dHx = - 2 * (x + u) * (1 - u) / r1**3 + 2 * (1 - u - x) * u / r2**3 + 2 * x
        dHy = - 2 *       y * (1 - u) / r1**3 - 2 *           y * u / r2**3 + 2 * y
        dHz = - 2 *       z * (1 - u) / r1**3 - 2 *           z * u / r2**3 
        dHdx = - 2 * dx
        dHdy = - 2 * dy
        dHdz = - 2 * dz

        return np.array( [ dHx, dHy, dHz, dHdx, dHdy, dHdz ] )

# Method to compute the gradient of the i2 term
def i2Gradient( dxdth2Section ):
    # Define J matrix
    matrixJ = np.zeros((6,6))
    matrixJ[0,1] = +2
    matrixJ[1,0] = -2
    matrixJ[0:3, 3:6] = -np.eye(3)
    matrixJ[3:6, 0:3] = np.eye(3)
    
    return matrixJ @ dxdth2Section

# CR3BP expanded dyanmical problem
def modelDynamicsCR3BPExp( state, dxdth2Section, l1, l2 ):
    x, y, z, dx, dy, dz = state[0], state[1], state[2], state[3], state[4], state[5]
    u = massParameter
    r1 = np.sqrt((u + x)**2 + y**2 + z**2)
    r2 = np.sqrt((1 - u - x)**2 + y**2 + z**2)

    ddx = x - (u + x) * (1 - u) / r1**3 + (1 - u - x) * u / r2**3 + 2 * dy
    ddy = y       - y * (1 - u) / r1**3           - y * u / r2**3 - 2 * dx
    ddz =         - z * (1 - u) / r1**3           - z * u / r2**3

    return np.array([dx, dy, dz, ddx, ddy, ddz]) + l1 * jacobiGradient(state) + l2 * i2Gradient(dxdth2Section)

# Method to expand a martix to N2*dimState
def expandMatrix( matrix, N ):
    dimState = 6
    # Expand rotation matrix to match dimensions N2 * dimState
    expandedRotationMatrix = np.zeros((N*dimState,N*dimState))
    for i in range(N):
            for j in range(N):
                iIndex = i*dimState
                jIndex = j*dimState
                expandedRotationMatrix[iIndex:iIndex+dimState,jIndex:jIndex+dimState] = \
                    np.eye(dimState) * matrix[i,j]

    return expandedRotationMatrix

# Method to compute rotation matrix
def getRotationMatrix( N, rho ):
    vect = []
    # Create rotation matrix
    for k in range(N):
        if k <= N//2 - 1:
            vect.append(np.exp(1j * 2 * np.pi * rho * k ))
        else:
            vect.append(np.exp(1j * 2 * np.pi * rho * (k-N) ))
    dft = fft(np.eye(N))
    rotationMatrix = np.linalg.inv(dft) @ (( np.eye(N).T * vect).T @ dft)
    return expandMatrix(rotationMatrix.real, N)
    
# Method to compute energy of the state
def getJacobi( state ):
    u = massParameter
    x, y, z = state[0], state[1], state[2]

    r_1 = np.sqrt((u + x)**2 + y**2 + z**2)
    r_2 = np.sqrt((1 - u - x)**2 + y**2 + z**2)
    U = (x**2 + y**2) + 2 * ((1 - u) / r_1 + u / r_2)
    V = np.linalg.norm(state[3:6])

    return U - V**2

# Method to compute the system of equations f(x,...)
def getCollocationSystem( torus, var, prevTorus, prevVar, tanTorus, tanVar, consts, N1, N2, m ):
    # Assign variables
    Href, dsref = consts
    T, rho, w1, w2, lambda1, lambda2 = var
    Tprev, rhoPrev, w1Prev, w2Prev, lambda1Prev, lambda2Prev = prevVar
    Ttan, rhoTan, _, _, _, _ = tanVar

    # Allocate vector
    dimState = 6
    dimVector = (N1 * (m + 1) + 1) * N2 * dimState
    scalarEqs = 6
    fArray = np.zeros(dimVector + scalarEqs)

    timeSegments = np.linspace(0, 1, N1 + 1)
    collocationArray = createGaussLegendreCollocationArray(N1,m)
    # Loop over  segments
    for numSegment in range(N1):
        # Method to compute derivative with interpolator
        def sumDerivative( torus, numAngle, time, timeSegment ):
            sol = np.zeros(dimState)
            for timeK in timeSegment[:-1]:
                sol += torus[timeK][numAngle] * lagrangePolDer(time, timeK, timeSegment)
            return sol

        # Time set (including extremes) of this segment
        timeSegment = [x for x in collocationArray if timeSegments[numSegment] <= x <= timeSegments[numSegment+1]]
        timeCollocation = timeSegment[1:-1]
        # Generate block for flow conditions (m+1) * N1 * N2 * dimState
        flowCondition = []
        for time in timeCollocation:
            dxdth2 = th2DerMatrix(N2) @ np.array(torus[time]).ravel()
            for numState, state in enumerate(torus[time]):
                indexState = dimState * numState
                dxdth2Section = dxdth2[indexState:indexState+dimState]
                flowCondition.append(sumDerivative(torus, numState, time, timeSegment) - T * (timeSegment[-1] - timeSegment[0]) * modelDynamicsCR3BPExp(state, dxdth2Section, lambda1, lambda2))

        # Generate block for continuity conditions
        contCondition = (np.sum(np.array([np.array(torus[timeK]) * lagrangePol(timeSegment[-1], timeK, timeSegment) for timeK in timeSegment[:-1]]), axis=0) - np.array(torus[timeSegment[-1]])).ravel()
        # Pack blocks
        flowIndex = numSegment * (m+1) * N2 * dimState
        contIndex = (numSegment+1) * (m+1) * N2 * dimState - N2 * dimState
        fArray[flowIndex:flowIndex+m*N2*dimState] = np.array(flowCondition).ravel()
        fArray[contIndex:contIndex+N2*dimState] = contCondition

    # Periodicity condition
    periodIndex = N1 * (m + 1) * N2 * dimState
    fArray[periodIndex:periodIndex+N2*dimState] = np.array(torus[0.0]).ravel() - getRotationMatrix(N2, -rho) @ np.array(torus[1.0]).ravel()
    
    # Scalar equations
    phaseCondition1 = 0
    phaseCondition2 = 0
    energyCondition = 0
    contCondition = 0
    for time in torus.keys():
        dxdth2 = th2DerMatrix(N2) @ np.array(prevTorus[time]).ravel()
        for numState, state in enumerate(torus[time]):
            indexState = dimState * numState
            dxdth2Section = dxdth2[indexState:indexState+dimState]
            dxdth1Section = Tprev * modelDynamicsCR3BPExp(state, dxdth2Section, lambda1Prev, lambda2Prev) - rhoPrev * dxdth2Section

            # Phase condition 1
            phaseCondition1 += np.dot(state - prevTorus[time][numState], dxdth1Section)
            # Phase condition 2
            phaseCondition2 += np.dot(state, dxdth2Section)
            # Energy condition  
            energyCondition += getJacobi(state)
            # Pseudo-arc length condition
            contCondition += np.dot(state - prevTorus[time][numState], tanTorus[time][numState])

    # Locate conditions
    fArray[dimVector] = phaseCondition1 / ((N1 * (m + 1) + 1) * N2)
    fArray[dimVector+1] = phaseCondition2 / ((N1 * (m + 1) + 1) * N2)
    fArray[dimVector+2] = energyCondition / ((N1 * (m + 1) + 1) * N2) - Href
    fArray[dimVector+3] = contCondition / ((N1 * (m + 1) + 1) * N2) - dsref

    # Extra relationship 1
    fArray[dimVector+4] = T * w1 - 1
    # Extra relationship 2
    fArray[dimVector+5] = T * w2 - rho

    return fArray

const = [3.00077, 0.001]
var = [T, rho, 1/T, rho/T, 0, 0]
fArray = getCollocationSystem(collocationTorus, var, collocationTorus, var, tanTorus, var, const, N1, N2, m)

# Method to compute jacobian of CR3BP
def getJacobianCR3BP( state ):
    x, y, z = state[0], state[1], state[2]
    u = massParameter
    r1 = np.sqrt((u + x)**2 + y**2 + z**2)
    r2 = np.sqrt((1 - u - x)**2 + y**2 + z**2)

    subBlock = np.zeros((6,6))
    subBlock[0,3] = 1
    subBlock[1,4] = 1
    subBlock[2,5] = 1

    subBlock[3,0] = 1 + 3 * (1 - u) * (x + u)**2 / r1**5 - (1 - u) / r1**3 + 3 * u * (1 - u - x)**2 / r2**5 - u / r2**3
    subBlock[3,1] = 3 * (1 - u) * (x + u) * y / r1**5 - 3 * u * (1 - u - x) * y / r2**5
    subBlock[3,2] = 3 * (1 - u) * (x + u) * z / r1**5 - 3 * u * (1 - u - x) * z / r2**5

    subBlock[4,0] = 3 * (1 - u) * (x + u) * y / r1**5 - 3 * u * (1 - u - x) * y / r2**5
    subBlock[4,1] = 1 + 3 * (1 - u) * y**2 / r1**5 - (1 - u) / r1**3 + 3 * u * y**2 / r2**5 - u / r2**3
    subBlock[4,2] = 3 * (1 - u) * y * z / r1**5 + 3 * u * y * z / r2**5

    subBlock[5,0] = 3 * (1 - u) * (x + u) * z / r1**5 - 3 * u * (1 - u - x) * z / r2**5
    subBlock[5,1] = 3 * (1 - u) * y * z / r1**5 + 3 * u * y * z / r2**5
    subBlock[5,2] = 3 * (1 - u) * z**2 / r1**5 - (1 - u) / r1**3 + 3 * u * z**2 / r2**5 - u / r2**3

    subBlock[3,4] = +2
    subBlock[4,3] = -2

    return subBlock

# Method to compute the derivative of the rotation matrix
def getRotMatDer( N, rho ):
    # Create rotation matrix
    vect = np.array([1j * k * 2 * np.pi * np.exp(1j * 2 * np.pi * rho * k ) for k in np.concatenate([np.arange(0,N//2), [0], np.arange(-N//2+1,0)])])
    dft = fft(np.eye(N))
    rotDer = np.linalg.inv(dft) @ ( np.eye(N).T * vect).T @ dft
    return expandMatrix(rotDer.real, N)

# Method to compute jacobian matrix
def getJacobianCollocation( torus, var, prevTorus, prevVar, tanTorus, tanVar, consts, N1, N2, m):
    # Assign variables
    Href, dsref = consts
    T, rho, w1, w2, lambda1, lambda2 = var
    Tprev, rhoPrev, w1Prev, w2Prev, lambda1Prev, lambda2Prev = prevVar
    Ttan, rhoTan, _, _, _, _ = tanVar

    # Allocate sparse matrix
    dimState = 6
    dimJacobian = (N1 * (m + 1) + 1) * N2 * dimState
    scalarEquation = 6
    jMatrix = lil_matrix((dimJacobian + scalarEquation, dimJacobian + scalarEquation))

    timeSegments = np.linspace(0, 1, N1 + 1)
    collocationArray = createGaussLegendreCollocationArray(N1, m)
    # Loop over  segments
    for numSegment in range(N1):
        # Time set (including extremes) of this segment
        timeSegment = [x for x in collocationArray if timeSegments[numSegment] <= x <= timeSegments[numSegment+1]]
        timeCollocation = timeSegment[1:-1] # times for which the flow condition is imposed

        dimBlock = (m + 1) * N2 * dimState
        segmentBlock = np.zeros((dimBlock, dimBlock+N2*dimState)) # extra column for continuity X_i+1
        periodDerivative = np.zeros(dimBlock)
        l1Derivative = np.zeros(dimBlock)
        l2Derivative = np.zeros(dimBlock)
        for numCollocation, time in enumerate(timeSegment):
            # Fill state derivative
            if time != timeSegment[-1]:
                colFillState = numCollocation * N2 * dimState
                for numFillState in range(m): 
                    rowFillState = numFillState * N2 * dimState
                    segmentBlock[rowFillState:rowFillState+N2*dimState, colFillState:colFillState+N2*dimState] = np.eye(N2 * dimState) * lagrangePolDer(timeSegment[numFillState+1], time, timeSegment) 

            # Generate flow derivatives only in collocation points
            if time in timeCollocation:
                circleDerivatives = [-(timeSegment[-1] - timeSegment[0]) * T * getJacobianCR3BP(state) for state in torus[time]]
                # Assing position
                for numState, stateMatrix in enumerate(circleDerivatives): 
                    rowFlowIndex = (numCollocation-1) * N2 * dimState + numState * dimState
                    colFlowIndex = rowFlowIndex + N2 * dimState
                    segmentBlock[rowFlowIndex:rowFlowIndex+dimState,colFlowIndex:colFlowIndex+dimState] += stateMatrix

                # Period derivative of flow equations
                periodDerRow = (numCollocation-1) * N2 * dimState
                dxdth2 = th2DerMatrix(N2) @ np.array(prevTorus[time]).ravel()
                periodDerivative[periodDerRow:periodDerRow+N2*dimState] = np.array([-(timeSegment[-1] - timeSegment[0]) * modelDynamicsCR3BPExp(state, dxdth2[dimState*numState:dimState*(numState+1)], lambda1, lambda2)
                                                                                     for numState, state in enumerate(torus[time])]).ravel()
                # Lambda 1 derivative
                l1Derivative[periodDerRow:periodDerRow+N2*dimState] = np.array([-(timeSegment[-1] - timeSegment[0]) * T * jacobiGradient(state) for state in torus[time]]).ravel()
                # Lambda 2 derivative
                l2Derivative[periodDerRow:periodDerRow+N2*dimState] = np.array([-(timeSegment[-1] - timeSegment[0]) * T * i2Gradient(dxdth2[dimState*numState:dimState*(numState+1)]) 
                                                                                    for numState, state in enumerate(torus[time])]).ravel()

            # Continuity condition
            rowContIndex = m * N2 * dimState
            colContIndex = numCollocation * N2 * dimState
            if time == timeSegment[-1]:
                continuityVal = -np.eye(N2*dimState)
            else:
                continuityVal = lagrangePol(timeSegment[-1], time, timeSegment) * np.eye(N2*dimState)
            segmentBlock[rowContIndex:rowContIndex+N2*dimState, colContIndex:colContIndex+N2*dimState] = continuityVal
        
        # Assign flow derivative wrt state blocks and period
        segmentIndex = numSegment * (m+1) * N2 * dimState
        heightSegment = (m+1) * N2 * dimState
        jMatrix[segmentIndex:segmentIndex+heightSegment, segmentIndex:segmentIndex+heightSegment+N2*dimState] = lil_matrix( segmentBlock )
        jMatrix[segmentIndex:segmentIndex+heightSegment, dimJacobian] = lil_matrix( periodDerivative.reshape(((m+1) * N2 * dimState, 1)) ) # period
        jMatrix[segmentIndex:segmentIndex+heightSegment, dimJacobian+4] = lil_matrix( l1Derivative.reshape(((m+1) * N2 * dimState, 1)) ) # lambda 1
        jMatrix[segmentIndex:segmentIndex+heightSegment, dimJacobian+5] = lil_matrix( l2Derivative.reshape(((m+1) * N2 * dimState, 1)) ) # lambda 2

    # Periodicity derivatives
    rowPeriodIndex = N1 * (m + 1) * N2 * dimState
    jMatrix[rowPeriodIndex:rowPeriodIndex+N2*dimState, 0:N2*dimState] = lil_matrix( np.eye(N2*dimState) ) # wrt X_0,0
    jMatrix[rowPeriodIndex:rowPeriodIndex+N2*dimState, rowPeriodIndex:rowPeriodIndex+N2*dimState] = lil_matrix( -1*getRotationMatrix(N2, -rho) ) # wrt X_N,0
    jMatrix[rowPeriodIndex:rowPeriodIndex+N2*dimState, dimJacobian+1] = lil_matrix( (-getRotMatDer(N2, -rho) @ np.array(torus[1.0]).ravel()).reshape((N2 * dimState, 1)) ) # wrt rho

    # Scalar equations derivatives
    for numTime, time in enumerate(torus.keys()):
        dxdth2 = th2DerMatrix(N2) @ np.array(prevTorus[time]).ravel()
        for numState, state in enumerate(torus[time]):
            indexState = dimState * numState
            dxdth2Section = dxdth2[indexState:indexState+dimState]
            dxdth1Section = Tprev * modelDynamicsCR3BPExp(state, dxdth2Section, lambda1Prev, lambda2Prev) - rhoPrev * dxdth2Section
            
            indexEl = numTime * N2 * dimState + numState * dimState
            # Phase condition 1 derivative
            jMatrix[dimJacobian, indexEl:indexEl+dimState] = dxdth1Section / ((N1 * (m + 1) + 1) * N2)
            # Phase condition 2 derivative
            jMatrix[dimJacobian+1, indexEl:indexEl+dimState] = dxdth2Section / ((N1 * (m + 1) + 1) * N2)
            # Energy condition derivatives
            jMatrix[dimJacobian+2, indexEl:indexEl+dimState] = jacobiGradient(state) / ((N1 * (m + 1) + 1) * N2)
            # Pseudo-arc length continuation derivatives
            jMatrix[dimJacobian+3, indexEl:indexEl+dimState] =  tanTorus[time][numState] / ((N1 * (m + 1) + 1) * N2)

    # Extra relationship 1 derivatives
    jMatrix[dimJacobian+4, dimJacobian:dimJacobian+4] = lil_matrix( np.array([w1, 0, T, 0]) )
    # Extra relationship 2 derivatives
    jMatrix[dimJacobian+5, dimJacobian:dimJacobian+4] = lil_matrix( np.array([w2, -1, 0, T]) )

    return csc_matrix(jMatrix)

jacobian = getJacobianCollocation( collocationTorus, var, collocationTorus, var, tanTorus, var, const, N1, N2, m)
if debugPlot:
    plt.spy(jacobian)
    plt.show()

# Method that creates a torus from a list of values
def createTorus(listVal, prevTorus, N2):
    solTorus = dict()
    count = 0
    for time in list(guessSol.keys()):
        solTorus.update({time:[]})
        for _ in range(N2):
            solTorus[time].append(listVal[count:count+6])
            count += 6
    return solTorus

rho += 1
Href = 3.000782035093267
ds = 0.0
prevSol = collocationTorus
guessSol = collocationTorus
const = [Href, ds]
prevVar = [T, rho, 1/T, rho/T, 0, 0]
var = prevVar

print("Test starting...")
fig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(10):
    print(i, guessSol[0.0][0])
    fArray = getCollocationSystem( guessSol, var, prevSol, prevVar, tanTorus, prevVar, const, N1, N2, m)
    jacobian = getJacobianCollocation( guessSol, var, prevSol, prevVar, tanTorus, prevVar, const, N1, N2, m)

    guessSolFlat = np.concatenate([np.array([state for time in guessSol.keys() for state in guessSol[time]]).ravel(), var])

    newGuess = guessSolFlat - spsolve(jacobian, fArray)
    var = newGuess[-6:]
    
    guessSol = createTorus(newGuess, guessSol, N2)
    printTorus(guessSol) 
plt.show()
    
