import numpy as np
from scipy.interpolate import griddata
from scipy.linalg import dft
from itertools import permutations
from scipy.special import roots_legendre
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt
from cr3bp import Cr3bp, PeriodicOrbit

fileName1 = "planarLyapunovNew.dat"
oneDay = 86400
distanceSunEarth = 149597870700 # in m
var = PeriodicOrbit("Sun","Earth", distanceSunEarth)
myDict = var.readDatabase(fileName1)
initialListPeriodic = list(myDict.values())[5]

nullTerminationCondition = lambda t,x: t + 1E3
nullTerminationCondition.terminal = True
nullTerminationCondition.direction = -1

initialTime = 0.0
initialStatePeriodic, halfPeriod = initialListPeriodic[0], initialListPeriodic[1]
# Retrieve history of data and state transition matrix
solData, stateTransition = var.runFullSimulationUntilTermination(initialStatePeriodic, initialTime, nullTerminationCondition, max_step=0.01, maxTime=2*halfPeriod)
timeSet = solData.t
# Retrieve the monodromy matrix

class QuasiPeriodic(Cr3bp):

    def __init__(self, massParameter):
        self.massParameter = massParameter

    def createSeedTorus( self, solutionData, stateTransition, N, offset ):
        monodromyMatrix = stateTransition[-1]
        eigenVector = self._getQuasiPeriodicEigenvector(monodromyMatrix)
        invariantCircleSeed = self._createSeedInvariantCircle(eigenVector, N , offset)
        return self._propagateFirstSeedInvariantCircle(invariantCircleSeed, stateTransition, solutionData)

    #Transforms a given torus to a collocation seed torus, that it with correct spacing
    def transformToCollocationSeedTorus( self, torus, N1, N2, m, percentageKeep=1):
        step = int(1/percentageKeep)
        # Create grid tau/theta 2 to interpolate
        time = np.asarray(list(torus.keys())[::step])
        angle = np.linspace(0, 1, N2, endpoint=False)
        gridInterpolate = [(a,b) for a in time for b in angle]

        # Create data to interpolate
        dataInterpolate = []
        for keyTime in time:
            for state in torus[keyTime]:
                dataInterpolate.append(state)

        # Create grid using Gauss-Legendre collocation points
        gaussLegrendreArray = self.createGaussLegendreCollocationArray(N1, m)
        xv, yv = np.meshgrid(gaussLegrendreArray, angle)
       
        # Create interpolator
        solInterpolator = griddata(gridInterpolate, dataInterpolate, (xv, yv), method='nearest')
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
        print(eigenValues)

        # If unitCircleVal empty raise error
        if not unitCircleVal:
            raise RuntimeError("The seed periodic orbit does not have an eigenvalue with quasi-periodic component. Change seed!")

        index = np.where(eigenValues == unitCircleVal)[0][0]
        return eigenVectors[:,index]

    # Create the first invariant circle to be used
    def _createSeedInvariantCircle( self, quasiPeriodicEigenVector, intervals, scalingFactor ):
        invariantCircle = []
        thetaSpan = np.linspace(0, 2*np.pi, intervals, endpoint=False)

        # Interate over the circle to create state vector at each element
        for angle in thetaSpan:
            stateAtAngle = np.cos(angle) * quasiPeriodicEigenVector.real - np.sin(angle) * quasiPeriodicEigenVector.imag
            invariantCircle.append(stateAtAngle * scalingFactor)

        return np.asarray(invariantCircle)

    # Propagates the first seed invariant circle using state transition matrices
    def _propagateFirstSeedInvariantCircle( self, firstInvariantCircle, setStateTransitionMatrices, solutionObject ):
        refPeriodicOrbit = solutionObject.y
        setTimes = solutionObject.t
        # Check that dimensions of the arrays are consistent
        if len(setStateTransitionMatrices) != len(setTimes):
            raise ValueError("Dimensions of times and set of state transition matrices must match!")

        seedTorus = dict()
        # The last transition matrix corresponds to the initial condition (so remove)
        for time, eachStateTransition in enumerate(setStateTransitionMatrices):
            setPropagatedStated = []
            for eachState in firstInvariantCircle:
                setPropagatedStated.append(eachStateTransition @ eachState)
            # Add reference state
            setPropagatedStated = np.asarray(setPropagatedStated)
            setPropagatedStated += refPeriodicOrbit[:,time]
            seedTorus.update({setTimes[time]/setTimes[-1]: setPropagatedStated})
        return seedTorus

    # Method to draw a torus
    def _printTorus( self, torus ):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for key in torus:
            circle = torus[key]
            ax.plot(circle[:,0], circle[:,1], circle[:,2], 'blue')
        plt.show()

    def computeF( self, guessTorus, previousSolution, tau1, tau2, T, rho, w1, w2, H, ds, N1, N2, m ):
        # Allocate vector
        fArray = np.empty(((N1 * (m + 1) + 1) * N2 * 6 + 6))
        timeSegments = np.linspace(0, 1, N1 + 1)
        collocationArray = self.createGaussLegendreCollocationArray(N1,m)

        # Iterate for times and fill array
        for numSegment in range(N1):
            timeSegment = [x for x in collocationArray if timeSegments[numSegment] <= x <= timeSegments[numSegment+1]]
            stateSegment = [guessTorus[x] for x in timeSegment]
            for i in range(m+2):
                time = timeSegment[i]
                for j, state in enumerate(guessTorus[time]):
                    fState = self.modelDynamicsCR3BP(0.0, state)
                    stateDerivative = self.computeDerivativeState(j, time, stateSegment, timeSegment)
                    fArray[i + 6*j: i + 6*j + 6] = stateDerivative - fState

        finalIndex = (N1 * (m + 1) + 1) * N2 * 6
        fArray[finalIndex-1:finalIndex+1] = self.getPhaseConditions(guessTorus, previousSolution)
        fArray[finalIndex+2] = self.getEnergyCondition(guessTorus, timeSegments, H)
        fArray[finalIndex+3] = self.getArcLengthCondition(guessTorus, previousSolution, ds)
        fArray[finalIndex+4] = T * w1 - 1
        fArray[finalIndex+5] = T * w2 - rho

        return fArray

    def getArcLengthCondition( self, guessTorus, previousSolution, ds ):
        # TODO: Complete
        return 0

    def getPhaseConditions( self, guessTorus, previousSolution):
        # TODO: Fill!!!
        phase1 = 0
        phase2 = 0
        
        return [phase1, phase2]

    def getEnergyCondition( self, guessTorus, timeSegments, H ):
        counter = 0
        energy = 0
        for i in range(len(timeSegments)):
            for j in guessTorus[timeSegments[i]]:
                counter += 1
                energy += self.getJacobi(j)

        return energy / counter - H

    def computeDerivativeState( self, angleElement, time, stateSegment, timeSegment ):
        ti = timeSegment[0]
        tii = timeSegment[-1]
        sol = 0
        for stateK, timeK in zip(stateSegment, timeSegment):
            sol += stateK[angleElement] * self.lagrangePolynomialDerivative(time, timeK, timeSegment)
        return sol * (tii - ti)
            
    def tauHat( self, t, ti, tii ):
        return (t - ti) / (tii - ti)

    def lagrangePolynomial( self, time, timeK, timeSegment):
        ti = timeSegment[0]
        tii = timeSegment[-1]
        sol = 1
        for timeJ in timeSegment[:-1]:
            if timeJ != timeK:
                sol *= (self.tauHat(time,ti,tii) - self.tauHat(timeJ,ti,tii)) / (self.tauHat(timeK,ti,tii) - self.tauHat(timeJ,ti,tii))
        return sol

    def lagrangePolynomialDerivative( self, time, timeK, timeSegment):
        ti = timeSegment[0]
        tii = timeSegment[-1]
        sol = 0
        for timeJ in timeSegment[:-1]:
            if timeJ != time:
                sol += 1 / (self.tauHat(time,ti,tii) - self.tauHat(timeJ,ti,tii))
        return sol * self.lagrangePolynomial(time, timeK, timeSegment)

    def computeJacobian( self, guessTorus, previousSolution, T, rho, w1, w2, N1, N2, m ):
        # Allocate sparse matrix
        dimState = 6
        dimJacobian = (N1 * (m + 1) + 1) * N2 * dimState
        scalarEquation = 2
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
                    
                    # Jacobian of each collocation point
                    rowCollocationState = numSegment * (m + 1) * N2 * dimState + numCollocation * N2 * dimState + numState * dimState
                    colCollocationState = rowCollocationState + N2 * dimState
                    derivativeTerm = (timeSegment[-1] - timeSegment[0]) * self.lagrangePolynomialDerivative(time, time, timeSegment)
                    jMatrix[rowCollocationState:rowCollocationState+dimState, colCollocationState:colCollocationState+dimState] = \
                        lil_matrix.astype( self.getJacobianState(state, derivativeTerm, T), dtype=np.double )

                    # Continuity conditions
                    rowContinuityCond = (numSegment + 1) * (N2 * m * dimState) + numSegment * N2 * dimState + numState * dimState
                    colFirstContinuityCond = numSegment * N2 * (m + 1) * dimState + numState * dimState

                    # Fill first continuity condition
                    if numCollocation == 0:
                        jMatrix[rowContinuityCond:rowContinuityCond+dimState, colFirstContinuityCond:colFirstContinuityCond+dimState] = \
                            lil_matrix.astype( np.eye(6) * self.lagrangePolynomial(1, time, timeSegment), dtype=np.double )
                    
                    # Fill continuity condition
                    jMatrix[rowContinuityCond:rowContinuityCond+dimState,colCollocationState:colCollocationState+dimState] = \
                            lil_matrix.astype(np.eye(6) * self.lagrangePolynomial(1, time, timeSegment), dtype=np.double )
                    
                    # Fill right column derivatives
                    # Period derivative 
                    jMatrix[rowCollocationState:rowCollocationState+dimState, dimJacobian] = -self.modelDynamicsCR3BP(0.0, state).reshape(6,1)

                    # Derivative wrt L1
                    # jMatrix[count:count+6,(N1 * (m + 1) + 1) * N2 * 6] = self.getHamiltonianDerivative(state).reshape(6,1)
                    # # Derivative wrt L2
                    # if j != len(circleStates)-1:
                    #     deltaTheta2 = (circleStates[j+1] - state) / N2
                    # else:
                    #     deltaTheta2 = (circleStates[0] - state) / N2
                    # jMatrix[count:count+6,(N1 * (m + 1) + 1) * N2 * 6 + 1] =  self.getI2Derivative(deltaTheta2).reshape(6,1)
                    # # Derivaive T
                    # 

                    # # Scalar equations dependent on the state # TODO: Complete
                    # # Phase condition 1
                    # jMatrix[depthScalarConditions,count:count+6] = self.getGradientPhase1(previousSolution)
                    # # Phase condition 2
                    # jMatrix[depthScalarConditions+1,count:count+6] = self.getGradientPhase2(previousSolution)
                    # # Energy condition
                    # jMatrix[depthScalarConditions+2,count:count+6] = self.getGradientHamiltonian(state)
                    # # Arc length
                    # jMatrix[depthScalarConditions+3,count:count+6] = self.getGradientArcLength()

                    # Update counter 

            # Equally spaced terms N1
            # Last continuity term
            rowContinuityCond = (numSegment + 1) * m * N2 * dimState + numSegment * N2 * dimState
            colContinuityCond = rowContinuityCond + N2 * dimState
            jMatrix[rowContinuityCond:rowContinuityCond + N2*dimState,colContinuityCond:colContinuityCond + N2*dimState] = \
                lil_matrix.astype( -np.eye(N2 * dimState), dtype=np.double )

            for numState, state in enumerate(guessTorus[timeSegment[-1]]):
                # Period derivative 
                jMatrix[rowContinuityCond+numState*dimState:rowContinuityCond+(1+numState)*dimState, dimJacobian] = \
                    -self.modelDynamicsCR3BP(0.0, state).reshape(6,1)
    
        # Periodicity condition
        rowPeriodicityCond = N1 * (m + 1) * N2 * dimState
        jMatrix[rowPeriodicityCond:rowPeriodicityCond+N2*dimState, 0:N2*dimState] = np.eye(N2*dimState)
        jMatrix[rowPeriodicityCond:rowPeriodicityCond+N2*dimState, rowPeriodicityCond:rowPeriodicityCond+N2*dimState] = \
            lil_matrix.astype( self.getRotationMatrix(rho,N2), dtype=np.double )

        # Rho derivative
        rotationMatrix =  np.linalg.inv(dft(N2)) @ (np.log(dft(N2)) @ np.power(dft(N2), -rho))
        jacobianRotationMatrix = np.zeros((N2*6,N2*6))
        for i in range(N2):
            for j in range(N2):
                jacobianRotationMatrix[i*6:i*6+6,j*6:j*6+6] = np.eye(6) * rotationMatrix[i,j].real
        
        stateLarge = np.array([])
        for state in guessTorus[timeSegment[-1]]:
            stateLarge = np.concatenate((stateLarge, state))
                
        jMatrix[rowPeriodicityCond:rowPeriodicityCond+N2*dimState, dimJacobian+1] = \
                    jacobianRotationMatrix @ np.array(stateLarge).reshape(N2*dimState,1)
                           
        # Phase conditions
        jMatrix[dimJacobian, 0:N2*dimState] = self.getPhaseCond1(guessTorus[0.0], previousSolution, N2)
        jMatrix[dimJacobian+1, 0:N2*dimState] = self.getPhaseCond2(guessTorus[0.0], previousSolution, N2)

        # Extra relationships
        # jMatrix[dimJacobian+2, dimJacobian:dimJacobian+4] = np.array([w1, 0, T, 0])
        # jMatrix[dimJacobian+3, dimJacobian:dimJacobian+4] = np.array([w2, -1, 0, T])

        # Convert to crc sparse matrix (improves operations)
        return csc_matrix(jMatrix)

    def getPhaseCond1( self, firstCircle, previousSolution, N2 ):
        keysPrevSol = list(previousSolution.keys())
        dt = keysPrevSol[1] - keysPrevSol[0]
        phaseCond1Array = np.zeros(N2*6)

        for numState in range(len(firstCircle)):
            prevTimeDerivative = (previousSolution[keysPrevSol[1]][numState] - previousSolution[keysPrevSol[0]][numState]) / dt
            phaseCond1Array[numState*6:numState*6+6] = prevTimeDerivative / len(firstCircle)

        return phaseCond1Array

    def getPhaseCond2( self, firstCircle, previousSolution, N2 ):
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

    def getRotationMatrix( self, rho, N2 ):
        # Creates rotation matrix
        rotationMatrix = np.linalg.inv(dft(N2)) @ np.power(dft(N2), -rho)

        jacobianRotationMatrix = np.zeros((N2*6,N2*6))
        for i in range(N2):
            for j in range(N2):
                jacobianRotationMatrix[i*6:i*6+6,j*6:j*6+6] = np.eye(6) * rotationMatrix[i,j].real

        return jacobianRotationMatrix

    def getGradientPhase1( self, previousTorus ):
        return 0

    def getGradientPhase2( self, previousTorus ):
        return 0

    def getGradientHamiltonian( self, state ):
        x, y, z, dx, dy, dz = state
        u = self.massParameter
        r1 = np.sqrt((u + x)**2 + y**2 + z**2)
        r2 = np.sqrt((1 - u - x)**2 + y**2 + z**2)

        hx = 2 * x - 2 * (x + u) * (1 - u) / r1**3 + 2 * (1 - u - x) * u / r2**3
        hy = 2 * y -       2 * y * (1 - u) / r1**3 -           2 * y * u / r2**3
        hz =       -       2 * z * (1 - u) / r1**3 -           2 * z * u / r2**3
        hdx = -2 * dx
        hdy = -2 * dy
        hdz = -2 * dz

        return np.array([hx, hy, hz, hdx, hdy, hdz])

    def getGradientArcLength( self ):
        return 0

    def getHamiltonianDerivative( self, stateGuess ):
        x, y, z, dx, dy, dz = stateGuess[0], stateGuess[1], stateGuess[2], stateGuess[3], stateGuess[4], stateGuess[5]
        u = self.massParameter
        r1 = np.sqrt((u + x)**2 + y**2 + z**2)
        r2 = np.sqrt((1 - u - x)**2 + y**2 + z**2)

        hx = 2 * x - 2 * (x + u) * (1 - u) / r1**3 + 2 * (1 - u - x) * u / r2**3
        hy = 2 * y -       2 * y * (1 - u) / r1**3 -           2 * y * u / r2**3
        hz =       -       2 * z * (1 - u) / r1**3 -           2 * z * u / r2**3
        hdx = -2 * dx
        hdy = -2 * dy
        hdz = -2 * dz

        return np.array([hx, hy, hz, hdx, hdy, hdz])

    def getI2Derivative( self, deltaTheta2 ):
        transMatrix = np.zeros((6,6))
        transMatrix[0,1] = 2
        transMatrix[1,0] = -2
        transMatrix[0:3,3:6] = -np.eye(3)
        transMatrix[3:6,0:3] = +np.eye(3)

        return transMatrix @ deltaTheta2

    def getJacobianState( self, stateGuess, derivativeTerm, period ):
        """
        Method that returns the (Jacobi) subblock at each collocation point.
        """
        x, y, z = stateGuess[0], stateGuess[1], stateGuess[2]
        u = self.massParameter
        r1 = np.sqrt((u + x)**2 + y**2 + z**2)
        r2 = np.sqrt((1 - u - x)**2 + y**2 + z**2)

        subBlock = np.eye(6) * derivativeTerm

        derivativeF = np.zeros((6,6))
        derivativeF[0,3] = derivativeF[1,4] = derivativeF[2,5] = 1
        derivativeF[3,4] = +2
        derivativeF[4,3] = -2

        derivativeF[3,0] = 1 + 3 * (1 - u) * (x + u)**2 / r1**5 - (1 - u) / r1**3 + 3 * u * (1 - u - x)**2 / r2**5 - u / r2**3
        derivativeF[3,1] = 3 * (1 - u) * (x + u) * y / r1**5 - 3 * u * (1 - u - x) * y / r2**5
        derivativeF[3,2] = 3 * (1 - u) * (x + u) * z / r1**5 - 3 * u * (1 - u - x) * z / r2**5

        derivativeF[4,0] = 3 * (1 - u) * (x + u) * y / r1**5 - 3 * u * (1 - u - x) * y / r2**5
        derivativeF[4,1] = 1 + 3 * (1 - u) * y**2 / r1**5 - (1 - u) / r1**3 + 3 * u * y**2 / r2**5 - u / r2**3
        derivativeF[4,2] = 3 * (1 - u) * y * z / r1**5 + 3 * u * y * z / r2**5

        derivativeF[5,0] = 3 * (1 - u) * (x + u) * z / r1**5 - 3 * u * (1 - u - x) * z / r2**5
        derivativeF[5,1] = 3 * (1 - u) * y * z / r1**5 + 3 * u * y * z / r2**5
        derivativeF[5,2] = 3 * (1 - u) * z**2 / r1**5 - (1 - u) / r1**3 + 3 * u * z**2 / r2**5 - u / r2**3

        return subBlock - period * derivativeF

N1 = 5
N2 = 5
m = 2
massParameter = 3.6E-6
foo = QuasiPeriodic(massParameter)
seedTorus = foo.createSeedTorus(solData, stateTransition, N2, 1E-3)
collocationTorus = foo.transformToCollocationSeedTorus(seedTorus, N1, N2, m, 0.2)
previousSolution = 0
sol = foo.computeF(collocationTorus, previousSolution, 0, 0, 0, 0, 0, 0, 3.12, 0.1, N1, N2, m)
jacobian = foo.computeJacobian(collocationTorus, seedTorus, 3.12, 1, 1/3.12, 1, N1, N2, m)
b = inv(jacobian)
plt.spy(b)
plt.show()
print("Test")