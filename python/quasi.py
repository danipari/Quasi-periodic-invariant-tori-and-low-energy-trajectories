import numpy as np
from scipy.interpolate import griddata
from itertools import permutations
from scipy.special import roots_legendre
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cr3bp import Cr3bp, PeriodicOrbit

def getSubBlock( stateGuess, massParameter, derivativeTerm, period ):
    """
    Method that returns the (Jacobi) subblock at each collocation point.
    """
    x, y, z = stateGuess[0], stateGuess[1], stateGuess[2]
    u = massParameter
    r1 = np.sqrt((u + x)**2 + y**2 + z**2)
    r2 = np.sqrt((1 - u - x)**2 + y**2 + z**2)

    subBlock = np.eye(6) * derivativeTerm
    subBlock[0,3] = subBlock[1,4] = subBlock[2,5] = -period
    subBlock[3,4] = -2 * period
    subBlock[4,3] = +2 * period

    subBlock[3,0] = 1 + 3 * (1 - u) * (x + u)**2 / r1**5 - (1 - u) / r1**3 + 3 * u * (1 - u - x)**2 / r2**5 - u / r2**3
    subBlock[3,1] = 3 * (1 - u) * (x + u) * y / r1**5 - 3 * u * (1 - u - x) * y / r2**5
    subBlock[3,2] = 3 * (1 - u) * (x + u) * z / r1**5 - 3 * u * (1 - u - x) * z / r2**5

    subBlock[4,0] = 3 * (1 - u) * (x + u) * y / r1**5 - 3 * u * (1 - u - x) * y / r2**5
    subBlock[4,1] = 1 + 3 * (1 - u) * y**2 / r1**5 - (1 - u) / r1**3 + 3 * u * y**2 / r2**5 - u / r2**3
    subBlock[4,2] = 3 * (1 - u) * y * z / r1**5 + 3 * u * y * z / r2**5

    subBlock[5,0] = 3 * (1 - u) * (x + u) * z / r1**5 - 3 * u * (1 - u - x) * z / r2**5
    subBlock[5,1] = 3 * (1 - u) * y * z / r1**5 + 3 * u * y * z / r2**5
    subBlock[5,2] = 3 * (1 - u) * z**2 / r1**5 - (1 - u) / r1**3 + 3 * u * z**2 / r2**5 - u / r2**3

    return subBlock


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
            for a in range(N2-1):
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

    def computeJacobian( self, guessTorus, N1, N2, m):
        # Allocate sparse matrix
        dim = (N1 * (m + 1) + 1) * N2 * 6 + 6
        jMatrix = lil_matrix((dim, dim))
        jMatrix[0:3,0:3] = np.eye(3)
        


N1 = 5
N2 = 5
m = 2
massParameter = 3.6E-6
foo = QuasiPeriodic(massParameter)
seedTorus = foo.createSeedTorus(solData, stateTransition, N2, 1E-3)
collocationTorus = foo.transformToCollocationSeedTorus(seedTorus, N1, N2, m, 0.2)
previousSolution = 0
sol = foo.computeF(collocationTorus, previousSolution, 0, 0, 0, 0, 0, 0, 3.12, 0.1, N1, N2, m)
foo.computeJacobian(collocationTorus, N1, N2, m)
print("Test")
