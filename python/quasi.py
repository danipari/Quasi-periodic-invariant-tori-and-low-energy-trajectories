import numpy as np
from scipy.interpolate import griddata
from scipy.linalg import dft
from scipy.fft import fft, ifft
from itertools import permutations
from scipy.special import roots_legendre
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt
from cr3bp import Cr3bp, PeriodicOrbit
from scipy.sparse.linalg import spsolve

class QuasiPeriodic(Cr3bp):

    def __init__(self, massParameter):
        self.massParameter = massParameter

    def createSeedTorus( self, solutionData, stateTransition, N, offset ):
        monodromyMatrix = stateTransition[-1]
        eigenValue, eigenVector = self._getQuasiPeriodicEigenvector(monodromyMatrix)
        rho = np.angle(eigenValue) / (2 * np.pi)
        invariantCircleSeed = self._createSeedInvariantCircle(eigenVector, N , offset)
        return self._propagateFirstSeedInvariantCircle(invariantCircleSeed, stateTransition, solutionData), rho

    #Transforms a given torus to a collocation seed torus, that it with correct spacing
    def transformToCollocationSeedTorus( self, torus, N1, N2, m, percentageKeep=1):
        step = int(1/percentageKeep)

        # Create grid tau/theta2 to interpolate
        keyTimes = list(torus.keys())
        # Should include last term for interpolation
        time = np.concatenate([np.array(keyTimes)[::step], [keyTimes[-1]]])
        timeNorm = time / keyTimes[-1]
        angle = np.linspace(0, 1, N2, endpoint=False)
        gridInterpolate = [(a,b) for a in timeNorm for b in angle]

        # Create data to interpolate
        dataInterpolate = []
        for keyTime in time:
            for state in torus[keyTime]:
                dataInterpolate.append(state)

        # Create grid using Gauss-Legendre collocation points
        gaussLegrendreArray = self.createGaussLegendreCollocationArray(N1, m)
        xv, yv = np.meshgrid(gaussLegrendreArray, angle)
       
        # Create interpolator
        solInterpolator = griddata(gridInterpolate, dataInterpolate, (xv, yv), method='linear')
        # Return corrected torus
        collocationTorus = dict()
        for i, t in enumerate(gaussLegrendreArray):
            circle = []
            for a in range(N2):
                circle.append(solInterpolator[a,i])
            collocationTorus.update({t: circle})

        return collocationTorus

    # Creates an array with the collocation values between [0,1]
    def createGaussLegendreCollocationArray( self, N, m ):
        collocationaArray = []

        for intervalValue in np.linspace(0, 1, N+1):
            collocationaArray.append(intervalValue)
            # Break the last iteration before filling
            if intervalValue == 1.0: break
            offset = intervalValue
            for root in roots_legendre(m)[0]:
                collocationaArray.append(offset + (root / 2.0 + 0.5) / N) # Tranform to the interval [0,1]

        return collocationaArray

    # Compute eigenvector within the unit circle modulus
    def _getQuasiPeriodicEigenvector( self, monodromyMatrix ):
        eigenValues = np.linalg.eig(monodromyMatrix)[0]
        eigenVectors = np.linalg.eig(monodromyMatrix)[1]
        unitCircleVal = eigenValues[(np.absolute(eigenValues) < 1) & (eigenValues.imag > 0)]

        # If unitCircleVal empty raise error
        if not unitCircleVal:
            raise RuntimeError("The seed periodic orbit does not have an eigenvalue with quasi-periodic component. Change seed!")

        index = np.where(eigenValues == unitCircleVal)[0][0]
        print("Eigenvalue selected:", eigenValues[index])
        return eigenValues[index], eigenVectors[:,index]

    # Create the first invariant circle to be used
    def _createSeedInvariantCircle( self, quasiPeriodicEigenVector, intervals, scalingFactor ):
        invariantCircle = []
        thetaSpan = np.linspace(0, 2*np.pi, intervals, endpoint=False)

        # Interate over the circle to create state vector at each element
        for angle in thetaSpan:
            stateAtAngle = np.cos(angle) * quasiPeriodicEigenVector.real - np.sin(angle) * quasiPeriodicEigenVector.imag
            invariantCircle.append(stateAtAngle * scalingFactor)

        return np.array(invariantCircle)

    # Propagates the first seed invariant circle using state transition matrices
    def _propagateFirstSeedInvariantCircle( self, firstInvariantCircle, setStateTransitionMatrices, solutionObject ):
        refPeriodicOrbit = solutionObject.y

        seedTorus = dict()
        # The last transition matrix corresponds to the initial condition (so remove)
        for i, (time, eachStateTransition) in enumerate(zip(solutionObject.t, setStateTransitionMatrices)):
            setPropagatedStated = []
            for eachState in firstInvariantCircle:
                setPropagatedStated.append(eachStateTransition @ eachState)
            # Add reference state
            setPropagatedStated = np.array(setPropagatedStated)
            setPropagatedStated += refPeriodicOrbit[:,i]
            seedTorus.update({time: setPropagatedStated})
        return seedTorus

    # Method to draw a torus
    def _printTorus( self, torus ):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for key in torus:
            circle = np.array(torus[key])
            ax.plot(circle[:,0], circle[:,1], circle[:,2], 'blue')
        plt.show()

    # LAGRANGE POLYNOMIAL DEFINITIONS v1.0
    def tauHat( self, t, ti, tii ):
        return (t - ti) / (tii - ti)

    def lagrangePolynomial( self, time, timeSampleK, timeSegment):
        ti = timeSegment[0]
        tii = timeSegment[-1]
        sol = 1
        for timeJ in timeSegment[:-1]:
            if timeJ != timeSampleK:
                sol *= (self.tauHat(time, ti, tii) - self.tauHat(timeJ, ti, tii)) / (self.tauHat(timeSampleK,ti,tii) - self.tauHat(timeJ,ti,tii))

        return sol

    def lagrangePolynomialDerivative( self, time, timeSampleK, timeSegment):
        ti = timeSegment[0]
        tii = timeSegment[-1]

        sol1 = 0
        for timeJ1 in timeSegment[:-1]:
            if timeJ1 != timeSampleK:
                sol2 = 1
                for timeJ2 in timeSegment[:-1]:
                    if timeJ2 != timeSampleK and timeJ2 != timeJ1:
                        sol2 *=  (self.tauHat(time, ti, tii) - self.tauHat(timeJ2, ti, tii)) / (self.tauHat(timeSampleK,ti,tii) - self.tauHat(timeJ2,ti,tii))
                sol2 *= 1. / (self.tauHat(timeSampleK,ti,tii) - self.tauHat(timeJ1,ti,tii))
                sol1 += sol2

        return sol1

    # SYSTEM OF EQNS CALCULATION
    def getCollocationSystem( self, guessTorus, prevTorus, T, rho, N1, N2, m ):
        # Allocate vector
        dimState = 6
        dimVector = (N1 * (m + 1) + 1) * N2 * dimState 
        scalarEqs = 0
        fArray = np.zeros(dimVector + scalarEqs)

        timeSegments = np.linspace(0, 1, N1 + 1)
        collocationArray = self.createGaussLegendreCollocationArray(N1,m)
        # Loop over  segments
        for numSegment in range(N1):
            timeSegment = [x for x in collocationArray if timeSegments[numSegment] <= x <= timeSegments[numSegment+1]]
            continuityVector = np.empty(N2 * dimState)  

            # Loop over N_i, collocation points and N_i+1
            for numCollocation in range(m+2):
                time = timeSegment[numCollocation]

                for numState in range(N2):
                    state = guessTorus[time][numState]

                    # Fill first entry (j=0) of the continuity vector
                    if numCollocation == 0:
                        indexCont = numState * dimState
                        continuityVector[indexCont:indexCont+dimState] = state * self.lagrangePolynomial(1.0, time, timeSegment)
                        continue
                    # Fill last entry (i+1) of the continuity vector
                    elif numCollocation == m + 1:
                        indexCont = numState * dimState
                        continuityVector[indexCont:indexCont+dimState] -= state
                        continue

                    # Fill flow equation in the collocation points
                    indexFlow = numSegment * ((m + 1) * N2 * dimState) + (numCollocation-1) * N2 * dimState + numState * dimState
                    fState = self.modelDynamicsCR3BP(0.0, state)
                    stateDerivative = state * self.lagrangePolynomialDerivative(time, time, timeSegment) * (timeSegment[-1] - timeSegment[0])
                    fArray[indexFlow:indexFlow+dimState] = stateDerivative - T * fState

                    # Fill continuity equation
                    indexCont = numState * dimState
                    continuityVector[indexCont:indexCont+dimState] += state * self.lagrangePolynomial(1.0, time, timeSegment) 

            # Fill the continuity equation in equally spaced points
            indexCont = (numSegment + 1) * (m * N2 * dimState) + numSegment * N2 * dimState
            fArray[indexCont:indexCont+N2*dimState] = continuityVector

        # Fill periodicity condition
        indexPeriod = N1 * (m + 1) * N2 * dimState
        fArray[indexPeriod:indexPeriod+N2*dimState] = self.getPeriodictyCond(guessTorus[timeSegment[0]], guessTorus[timeSegment[-1]], rho)

        # # Fill phase condition 1
        # fArray[dimVector] = self.getPhaseCond1(guessTorus[timeSegment[0]], prevTorus)
        # # Fill phase condition 2
        # fArray[dimVector+1] = self.getPhaseCond2(guessTorus[timeSegment[0]], prevTorus)

        return fArray

    def getPhaseCond1( self, firstCircle, prevTorus ):
        prevFirstCircle = prevTorus[0.0]
        keysPrevTorus = list(prevTorus.keys())
        time0 = keysPrevTorus[0]
        time1 = keysPrevTorus[1]

        sol = 0
        for numState, (state, prevState) in enumerate(zip(firstCircle, prevFirstCircle)):
            prevTimeDerivative = (prevTorus[time1][numState] - prevTorus[time0][numState]) / (time1 - time0)
            sol += np.dot(state - prevState, prevTimeDerivative)

        return sol / len(firstCircle)

    def getPhaseCond2( self, firstCircle, prevTorus ):
        prevFirstCircle = prevTorus[0.0]
        dtheta = 1.0 / len(firstCircle)

        sol = 0
        for numState, (state, prevState) in enumerate(zip(firstCircle, prevFirstCircle)):
            nextPrevState = prevFirstCircle[0] if numState == len(prevFirstCircle) - 1 else prevFirstCircle[numState+1]
            prevThetaDerivative = (nextPrevState - prevState) / dtheta
            sol += np.dot(state - prevState, prevThetaDerivative)

        return sol / len(firstCircle)     

    def getPeriodictyCond( self, firstCircle, lastCircle, rho ):
        dimState = 6
        firstVector = np.empty(N2*dimState)
        lastVector = np.empty(N2*dimState)
        for numState, (firstState, lastState) in enumerate(zip(firstCircle, lastCircle)):
            index = numState * dimState
            firstVector[index:index+dimState] = firstState
            lastVector[index:index+dimState] = lastState

        rotMatrix = self.getRotationMatrix(rho, N2)
        return firstVector - rotMatrix @ lastVector

    def computeDerivativeState( self, angleElement, time, stateSegment, timeSegment ):
        ti = timeSegment[0]
        tii = timeSegment[-1]
        sol = 0
        for stateK, timeK in zip(stateSegment, timeSegment):
            sol += stateK[angleElement] * self.lagrangePolynomialDerivative(time, timeK, timeSegment)
        return sol * (tii - ti)

    # JACOBIAN CALCULATION
    def computeJacobian( self, guessTorus, previousSolution, T, rho, N1, N2, m ):
        # Allocate sparse matrix
        dimState = 6
        dimJacobian = (N1 * (m + 1) + 1) * N2 * dimState
        scalarEquation = 0
        jMatrix = lil_matrix((dimJacobian + scalarEquation, dimJacobian + scalarEquation))

        timeSegments = np.linspace(0, 1, N1 + 1)
        collocationArray = self.createGaussLegendreCollocationArray(N1,m)
        
        # Iterate for times and fill array
        for numSegment in range(N1):
            timeSegment = [x for x in collocationArray if timeSegments[numSegment] <= x <= timeSegments[numSegment+1]]

            # Collocation points
            for numCollocation in range(m):
                time = timeSegment[numCollocation+1]

                for numState, state in enumerate(guessTorus[time]):
                    
                    ## Flow condition
                    # Derivative wrt states
                    rowCollocationState = numSegment * (m + 1) * N2 * dimState + numCollocation * N2 * dimState + numState * dimState
                    colCollocationState = rowCollocationState + N2 * dimState
                    derivativeTerm = self.lagrangePolynomialDerivative(time, time, timeSegment) / (timeSegment[-1] - timeSegment[0])
                    jMatrix[rowCollocationState:rowCollocationState+dimState, colCollocationState:colCollocationState+dimState] = \
                        lil_matrix.astype( self.getJacobianFlow(state, derivativeTerm, T), dtype=np.double )

                    # Derivative wrt period
                    # jMatrix[rowCollocationState:rowCollocationState+dimState, dimJacobian] = -self.modelDynamicsCR3BP(0.0, state).reshape(6,1)

                    ## Continuity conditions
                    rowContinuityCond = (numSegment + 1) * (N2 * m * dimState) + numSegment * N2 * dimState + numState * dimState
                    colFirstContinuityCond = numSegment * N2 * (m + 1) * dimState + numState * dimState

                    # Fill first continuity condition
                    if numCollocation == 0:
                        jMatrix[rowContinuityCond:rowContinuityCond+dimState, colFirstContinuityCond:colFirstContinuityCond+dimState] = \
                            lil_matrix.astype( np.eye(6) * self.lagrangePolynomial(1, time, timeSegment), dtype=np.double )
                    
                    # Fill continuity condition
                    jMatrix[rowContinuityCond:rowContinuityCond+dimState,colCollocationState:colCollocationState+dimState] = \
                            lil_matrix.astype(np.eye(6) * self.lagrangePolynomial(1, time, timeSegment), dtype=np.double )
                    
            # Equally spaced terms N1
            # Last continuity term
            rowContinuityCond = (numSegment + 1) * m * N2 * dimState + numSegment * N2 * dimState
            colContinuityCond = rowContinuityCond + N2 * dimState
            jMatrix[rowContinuityCond:rowContinuityCond + N2*dimState,colContinuityCond:colContinuityCond + N2*dimState] = \
                lil_matrix.astype( -np.eye(N2 * dimState), dtype=np.double )

            # for numState, state in enumerate(guessTorus[timeSegment[-1]]):
            #     # Period derivative 
            #     jMatrix[rowContinuityCond+numState*dimState:rowContinuityCond+(1+numState)*dimState, dimJacobian] = \
            #         -self.modelDynamicsCR3BP(0.0, state).reshape(6,1)
    
        # Periodicity condition
        rowPeriodicityCond = N1 * (m + 1) * N2 * dimState
        jMatrix[rowPeriodicityCond:rowPeriodicityCond+N2*dimState, 0:N2*dimState] = np.eye(N2*dimState)
        jMatrix[rowPeriodicityCond:rowPeriodicityCond+N2*dimState, rowPeriodicityCond:rowPeriodicityCond+N2*dimState] = \
            lil_matrix.astype( -self.getRotationMatrix(rho,N2), dtype=np.double )

        # Rho derivative       
        # fullStateLastEl = np.concatenate(guessTorus[1.0]).ravel()
        # jMatrix[rowPeriodicityCond:rowPeriodicityCond+N2*dimState, dimJacobian+1] = \
        #             -self.getRotationMatrix(rho,N2,True) @ fullStateLastEl.reshape(N2*dimState,1)
                           
        # Phase conditions
        # jMatrix[dimJacobian, 0:N2*dimState] = self.getPhaseCond1Der(guessTorus[0.0], previousSolution, N2)
        # jMatrix[dimJacobian+1, 0:N2*dimState] = self.getPhaseCond2Der(guessTorus[0.0], previousSolution, N2)

        # Convert to crc sparse matrix (improves operations)
        return csc_matrix(jMatrix)

    def getPhaseCond1Der( self, firstCircle, previousSolution, N2 ):
        keysPrevSol = list(previousSolution.keys())
        dt = keysPrevSol[1] - keysPrevSol[0]
        phaseCond1Array = np.zeros(N2*6)

        for numState in range(len(firstCircle)):
            prevTimeDerivative = (previousSolution[keysPrevSol[1]][numState] - previousSolution[keysPrevSol[0]][numState]) / dt
            phaseCond1Array[numState*6:numState*6+6] = prevTimeDerivative / len(firstCircle)

        return phaseCond1Array

    def getPhaseCond2Der( self, firstCircle, previousSolution, N2 ):
        phaseCond1Array = np.zeros(N2*6)
        dtheta = 1 / N2

        for numState, state in enumerate(firstCircle):
            if numState == len(firstCircle) - 1:
                nextState = firstCircle[0]
            else:
                nextState = firstCircle[numState+1]

            prevTimeDerivative = (nextState - state) / dtheta
            phaseCond1Array[numState*6:numState*6+6] = prevTimeDerivative / len(firstCircle)

        return phaseCond1Array     

    def getRotationMatrix( self, rho, N2, returnDerivative=False ):
        # TODO: Organize
        def createRotationMatrix(N):
            vect = []
            for k in range(N):
                if k <= N/2 - 1:
                    vect.append(np.exp(-2 * np.pi * 1j * rho * k ))
                else:
                    vect.append(np.exp(-2 * np.pi * 1j * rho * (k-N) ))
            dft = fft(np.eye(N))
            rotationMatrix = np.linalg.inv(dft) @ (( np.eye(N).T * vect).T @ dft)
            return rotationMatrix.real

        def createRotationMatrixDerivative(N):
            vect = []
            for k in range(N):
                if k <= N/2 - 1:
                    vect.append(-2 * np.pi * 1j * k * np.exp(-2 * np.pi * 1j * rho * k ))
                else:
                    vect.append(-2 * np.pi * 1j * (k-N) * np.exp(-2 * np.pi * 1j * rho * (k-N) ))
            dft = fft(np.eye(N))
            rotationMatrix = np.linalg.inv(dft) @ (( np.eye(N).T * vect).T @ dft)
            return rotationMatrix.real

        dimState = 6
        # Creates rotation matrix (with shift -rho)
        if returnDerivative:
            rotationMatrix = createRotationMatrix(N2)
        else:
            rotationMatrix = createRotationMatrixDerivative(N2)

        jacobianRotationMatrix = np.zeros((N2*dimState,N2*dimState))
        for i in range(N2):
            for j in range(N2):
                iIndex = i*dimState
                jIndex = j*dimState
                jacobianRotationMatrix[iIndex:iIndex+dimState,jIndex:jIndex+dimState] = \
                    np.eye(dimState) * rotationMatrix[i,j].real

        return jacobianRotationMatrix

    def getJacobianFlow( self, stateGuess, derivativeTerm, T ):
        """
        Method that returns the (Jacobi) subblock at each collocation point.
        """
        x, y, z = stateGuess[0], stateGuess[1], stateGuess[2]
        u = self.massParameter
        r1 = np.sqrt((u + x)**2 + y**2 + z**2)
        r2 = np.sqrt((1 - u - x)**2 + y**2 + z**2)

        subBlock = np.eye(6) * derivativeTerm

        derivativeF = np.zeros((6,6))
        derivativeF[0,3] = 1
        derivativeF[1,4] = 1
        derivativeF[2,5] = 1

        derivativeF[3,0] = 1 + 3 * (1 - u) * (x + u)**2 / r1**5 - (1 - u) / r1**3 + 3 * u * (1 - u - x)**2 / r2**5 - u / r2**3
        derivativeF[3,1] = 3 * (1 - u) * (x + u) * y / r1**5 - 3 * u * (1 - u - x) * y / r2**5
        derivativeF[3,2] = 3 * (1 - u) * (x + u) * z / r1**5 - 3 * u * (1 - u - x) * z / r2**5

        derivativeF[4,0] = 3 * (1 - u) * (x + u) * y / r1**5 - 3 * u * (1 - u - x) * y / r2**5
        derivativeF[4,1] = 1 + 3 * (1 - u) * y**2 / r1**5 - (1 - u) / r1**3 + 3 * u * y**2 / r2**5 - u / r2**3
        derivativeF[4,2] = 3 * (1 - u) * y * z / r1**5 + 3 * u * y * z / r2**5

        derivativeF[5,0] = 3 * (1 - u) * (x + u) * z / r1**5 - 3 * u * (1 - u - x) * z / r2**5
        derivativeF[5,1] = 3 * (1 - u) * y * z / r1**5 + 3 * u * y * z / r2**5
        derivativeF[5,2] = 3 * (1 - u) * z**2 / r1**5 - (1 - u) / r1**3 + 3 * u * z**2 / r2**5 - u / r2**3

        derivativeF[3,4] = +2
        derivativeF[4,3] = -2

        return subBlock - T * derivativeF


fileName1 = "planarLyapunovNew.dat"
oneDay = 86400
distanceSunEarth = 149597870700 # in m
var = PeriodicOrbit("Sun","Earth", distanceSunEarth)
myDict = var.readDatabase(fileName1)
initialListPeriodic = list(myDict.values())[7]

nullTerminationCondition = lambda t,x: t + 1E3
nullTerminationCondition.terminal = True
nullTerminationCondition.direction = -1

initialTime = 0.0
initialStatePeriodic, halfPeriod = initialListPeriodic[0], initialListPeriodic[1]
# Retrieve history of data and state transition matrix
solData, stateTransition = var.runFullSimulationUntilTermination(initialStatePeriodic, initialTime, nullTerminationCondition, max_step=0.01, maxTime=2*halfPeriod)

N1 = 50
N2 = 50
m = 5
massParameter = 3.003480642487067e-06
foo = QuasiPeriodic(massParameter)
seedTorus, rho = foo.createSeedTorus(solData, stateTransition, N2, 0.1)
collocationTorus = foo.transformToCollocationSeedTorus(seedTorus, N1, N2, m, 1)
# foo._printTorus(collocationTorus)
# index = 10
# foo._printTorus({0.0: collocationTorus[list(collocationTorus.keys())[index]]})
sol = foo.getCollocationSystem(collocationTorus, seedTorus, 3.12, rho, N1, N2, m)
jacobian = foo.computeJacobian(collocationTorus, seedTorus, 3.12, rho, N1, N2, m)
x = spsolve(jacobian, sol)
plt.spy(jacobian)
plt.show()
# print("Test")

def createTorus(listVal, prevTorus, N2):
    solTorus = dict()
    count = 0
    for time in list(guessSol.keys()):
        solTorus.update({time:[]})
        for _ in range(N2):
            solTorus[time].append(listVal[count:count+6])
            count += 6
    return solTorus

guessSol = collocationTorus
prevSol = collocationTorus
T = 1.5063 * 2
for i in range(10):
    print(i, guessSol[0.0][0])
    f = foo.getCollocationSystem(guessSol, prevSol, T, rho, N1, N2, m)
    jacobian = foo.computeJacobian(guessSol, prevSol, T, rho, N1, N2, m)
    guessSolFlat = np.concatenate([np.concatenate(list(guessSol.values())).ravel()])

    newGuess = guessSolFlat - spsolve(jacobian, f)
    guessSol = createTorus(newGuess, guessSol, N2)
    