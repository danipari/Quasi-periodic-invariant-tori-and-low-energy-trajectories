#!/usr/bin/env python

import enum
import numpy as np
from odeintw import odeintw
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.integrate as integrate
from mpl_toolkits.mplot3d import Axes3D

class Cr3bp:

    class LagrangePoint(enum.Enum):
        l1 = 1; l2 = 2; l3 = 3; l4 = 4; l5 = 5     

    gravitationalParameter = {
        "Sun": 1.32712440018E20,
        "Earth": 3.986004418E14 }

    def __init__( self, bodyPrimary, bodySecondary, distanceBodies ):
        try :
            self.gravParameterPrimary = self.gravitationalParameter[bodyPrimary]
            self.gravParameterSecondary = self.gravitationalParameter[bodySecondary]
        except:
            raise KeyError("Some of the bodies has not a gravitational parameter value associated.")
        
        self.massParameter = self.gravParameterSecondary / (self.gravParameterPrimary + self.gravParameterSecondary)
        self.distanceBodies = distanceBodies

    def getJacobi( self, state ):
        x, y, z = state[0], state[1], state[2]

        r_1 = np.sqrt((self.massParameter + x)**2 + y**2 + z**2)
        r_2 = np.sqrt((1 - self.massParameter - x)**2 + y**2 + z**2)
        U = (x**2 + y**2) + 2 * ((1 - self.massParameter) / r_1 + self.massParameter / r_2)
        V = np.linalg.norm(state[3:6])

        return U - V**2

    def getLagrangePoint( self, lagrangePoint, tol=1E-12, maxiter=50 ):
        u = self.massParameter
        stateLagrange = np.zeros(6)
        
        # Collinear points
        if lagrangePoint == self.LagrangePoint.l1:
            rootFunL1 = lambda x : x - (1 - u) / (u + x)**2 + u / (1 - u - x)**2
            rootFunL1Derivative = lambda x : 1 + 2 * (1 - u) / (u + x)**3 + 2 * u / (1 - u - x)**3

            xRoot = optimize.newton(rootFunL1, 1, rootFunL1Derivative, tol=tol, maxiter=maxiter)
            stateLagrange[0] = xRoot

        elif lagrangePoint == self.LagrangePoint.l2:
            rootFunL2 = lambda x : x - (1 - u) / (u + x)**2 - u / (1 - u - x)**2
            rootFunL2Derivative = lambda x : 1 + 2 * (1 - u) / (u + x)**3 - 2 * u / (1 - u - x)**3

            xRoot = optimize.newton(rootFunL2, 1, rootFunL2Derivative, tol=tol, maxiter=maxiter)
            stateLagrange[0] = xRoot

        elif lagrangePoint == self.LagrangePoint.l3:
            rootFunL3 = lambda x : x + (1 - u) / (u + x)**2 - u / (1 - u - x)**2
            rootFunL3Derivative = lambda x : 1 - 2 * (1 - u) / (u + x)**3 - 2 * u / (1 - u - x)**3

            xRoot = optimize.newton(rootFunL3, -1, rootFunL3Derivative, tol=tol, maxiter=maxiter)
            stateLagrange[0] = xRoot
            
        # Equilateral points
        elif (lagrangePoint in [self.LagrangePoint.l4, self.LagrangePoint.l5]):
            raise ValueError("L3 and L4 Lagrange points not implemented yet!") # TODO: Add L3/4 Lagrange point

        return stateLagrange

    def modelDynamicsCR3BP( self, t, state ):
        x, y, z, dx, dy, dz = state[0], state[1], state[2], state[3], state[4], state[5]
        u = self.massParameter
        r1 = np.sqrt((u + x)**2 + y**2 + z**2)
        r2 = np.sqrt((1 - u - x)**2 + y**2 + z**2)

        ddx = x - (u + x) * (1 - u) / r1**3 + (1 - u - x) * u / r2**3 + 2 * dy
        ddy = y       - y * (1 - u) / r1**3           - y * u / r2**3 - 2 * dx
        ddz =         - z * (1 - u) / r1**3           - z * u / r2**3

        return np.array([dx, dy, dz, ddx, ddy, ddz])

    def stateTransitionDerivative( self, state, phi ):
        u = self.massParameter
        x, y, z = state[0], state[1], state[2]
        r1 = np.sqrt((u + x)**2 + y**2 + z**2)
        r2 = np.sqrt((1 - u - x)**2 + y**2 + z**2)

        A = np.zeros((6,6))
        A[0,3] = 1
        A[1,4] = 1
        A[2,5] = 1

        A[3,0] = 1 + 3 * (1 - u) * (x + u)**2 / r1**5 - (1 - u) / r1**3 + 3 * u * (1 - u - x)**2 / r2**5 - u / r2**3
        A[3,1] = 3 * (1 - u) * (x + u) * y / r1**5 - 3 * u * (1 - u - x) * y / r2**5
        A[3,2] = 3 * (1 - u) * (x + u) * z / r1**5 - 3 * u * (1 - u - x) * z / r2**5

        A[4,0] = 3 * (1 - u) * (x + u) * y / r1**5 - 3 * u * (1 - u - x) * y / r2**5
        A[4,1] = 1 + 3 * (1 - u) * y**2 / r1**5 - (1 - u) / r1**3 + 3 * u * y**2 / r2**5 - u / r2**3
        A[4,2] = 3 * (1 - u) * y * z / r1**5 + 3 * u * y * z / r2**5

        A[5,0] = 3 * (1 - u) * (x + u) * z / r1**5 - 3 * u * (1 - u - x) * z / r2**5
        A[5,1] = 3 * (1 - u) * y * z / r1**5 + 3 * u * y * z / r2**5
        A[5,2] = 3 * (1 - u) * z**2 / r1**5 - (1 - u) / r1**3 + 3 * u * z**2 / r2**5 - u / r2**3

        A[3,4] = 2
        A[4,3] = -2

        dF = A @ phi

        return dF

    def runSimulation( self, initialState, tSpan, max_step=1 ):
        return integrate.solve_ivp(self.modelDynamicsCR3BP, tSpan, initialState, max_step=max_step)         

    def timeDimensionalToNormalized( self, dimensionalTime ):
        return dimensionalTime * np.sqrt((self.gravParameterPrimary + self.gravParameterSecondary) / self.distanceBodies**3 ) # TODO: Implement the other transformations


class PeriodicOrbit(Cr3bp):

    class OrbitFamily( enum.Enum ):
        planarLyapunov = 1; verticalLyapunov = 2; northHalo = 3; southHalo = 4

    def runFullSimulationUntilTermination( self, initialState, initialTime, terminationFunction, max_step=1, maxTime=5000.0 ):      
        tSpan = [initialTime, maxTime]
        integrator = integrate.solve_ivp(self.modelDynamicsCR3BP, tSpan, initialState, max_step=max_step, events=terminationFunction, dense_output=True) 
        stateTransFun = lambda y, t : self.stateTransitionDerivative(integrator.sol(t), y) 
        stateTransitionSolution = odeintw(stateTransFun, np.eye(6), [initialTime, integrator.t[-1]])
                
        return integrator, stateTransitionSolution[-1]

    def applyCorrectorPredictor( self, approxInitialState, initialTime, lagrangePoint, orbitFamily, tol=1E-8, returnTerminatingPeriod=False ):
        # Initialize elements
        initialState = approxInitialState
        error = 1

        # Select termination function according to the family
        if orbitFamily == self.OrbitFamily.verticalLyapunov:
            terminationCondition = lambda t,x: x[2]
        else:
            terminationCondition = lambda t,x: x[1]
        terminationCondition.terminal = True
        terminationCondition.direction = -1

        while abs(error) > tol:
            stateEvolution, stateTransition = self.runFullSimulationUntilTermination(initialState, initialTime, terminationCondition, max_step=0.01)
            finalState = stateEvolution.y[:,-1]

            if orbitFamily == self.OrbitFamily.planarLyapunov :
                error = abs(finalState[3])
                initialState[4] -= finalState[3] / stateTransition[3,4]
    
            elif orbitFamily in [self.OrbitFamily.northHalo, self.OrbitFamily.southHalo]: # TODO: functionize
                # Z is kept constant only x and dy change
                xDotError, zDotError = -finalState[3], -finalState[5]
                error = np.sqrt(xDotError**2 + zDotError**2)
                finalStateDeriv = self.modelDynamicsCR3BP(0.0, finalState)
                
                smallPhi = np.zeros((2,2))
                smallPhi[0,0] = stateTransition[3,0] - stateTransition[1,0] * finalStateDeriv[3] / finalStateDeriv[1] 
                smallPhi[0,1] = stateTransition[3,4] - stateTransition[1,4] * finalStateDeriv[3] / finalStateDeriv[1] 
                smallPhi[1,0] = stateTransition[5,0] - stateTransition[1,0] * finalStateDeriv[5] / finalStateDeriv[1] 
                smallPhi[1,1] = stateTransition[5,4] - stateTransition[1,4] * finalStateDeriv[5] / finalStateDeriv[1]

                xCorrection, yDotCorrection = np.linalg.solve(smallPhi, np.array([xDotError, zDotError]))
                initialState[0] += xCorrection
                initialState[4] += yDotCorrection

            elif orbitFamily == self.OrbitFamily.verticalLyapunov:
                yError, xDotError = -finalState[1], -finalState[3]
                error = np.sqrt(xDotError**2 + yError**2)

                # Z is kept constant only x and dy change
                finalStateDeriv = self.modelDynamicsCR3BP(0.0, finalState) 
                smallPhi = np.zeros((2,2))
                smallPhi[0,0] = stateTransition[1,0] - stateTransition[2,0] * finalStateDeriv[1] / finalStateDeriv[2] 
                smallPhi[0,1] = stateTransition[1,4] - stateTransition[2,4] * finalStateDeriv[1] / finalStateDeriv[2] 
                smallPhi[1,0] = stateTransition[3,0] - stateTransition[2,0] * finalStateDeriv[3] / finalStateDeriv[2]
                smallPhi[1,1] = stateTransition[3,4] - stateTransition[2,4] * finalStateDeriv[3] / finalStateDeriv[2] 

                xCorrection, yDotCorrection = np.linalg.solve(smallPhi, np.array([yError, xDotError ]))
                initialState[0] += xCorrection
                initialState[4] += yDotCorrection

        if returnTerminatingPeriod:
            return initialState, stateEvolution.t[-1]
        else:
            return initialState

    def getApproximateInitialState( self, familyParameter, lagrangePoint, orbitFamily ):
        if (orbitFamily == self.OrbitFamily.planarLyapunov):
            return self._approxInitialStatePlanar(familyParameter, lagrangePoint)
        else:
            return self._approxInitialState3D(familyParameter, lagrangePoint, orbitFamily)

    def _approxInitialState3D( self, Az, lagrangePoint, orbitFamily ):
        u = self.massParameter
        distLagrange = self.getLagrangePoint(lagrangePoint)[0]
        gamma = (1 - u) - distLagrange   
        c_2 = (u + (1 - u) * gamma**3 / (1 - gamma)**3) / gamma**3
        c_3 = (u - (1 - u) * gamma**4 / (1 - gamma)**4) / gamma**3
        c_4 = (u + (1 - u) * gamma**5 / (1 - gamma)**5) / gamma**3

        lmda = np.sqrt(((2 - c_2) + np.sqrt((c_2 - 2)**2 + 4 * (c_2 - 1) * (1 + 2 * c_2))) / 2)
        # In the vertical case Ax == Ay so k = 1
        k = 1 if orbitFamily == self.OrbitFamily.verticalLyapunov else 2 * lmda / (lmda**2 + 1 - c_2)

        d_1 = 3 * lmda**2 / k * (k * (6 * lmda**2 - 1) - 2 * lmda)
        d_2 = 8 * lmda**2 / k * (k * (11 * lmda**2 - 1) - 2 * lmda)

        a_21 = 3 * c_3 * (k**2 - 2) / (4 * (1 + 2 * c_2))
        a_22 = 3 * c_3 / (4 * (1 + 2 * c_2))
        a_23 = -3 * c_3 * lmda / (4 * k * d_1) * (3 * k**3 * lmda - 6 * k * (k - lmda) + 4)
        a_24 = -3 * c_3 * lmda / (4 * k * d_1) * (2 + 3 * k * lmda)
        b_21 = -3 * c_3 * lmda / (2 * d_1) * (3 * k * lmda - 4)
        b_22 = 3 * c_3 * lmda / d_1
        d_21 = -c_3 / (2 * lmda**2)

        d_31 = 3 / (64 * lmda**2) * (4 * c_3 * a_24 + c_4)
        a_31 = -9 * lmda / (4 * d_2) * (4 * c_3 * (k * a_23 - b_21) + k * c_4 * (4 + k**2)) + \
            (9 * lmda**2 + 1 - c_2) / (2 * d_2) * (3 * c_3 * (2 * a_23 - k * b_21) + c_4 * (2 + 3 * k**2))
        a_32 = -1/d_2 * (9 * lmda / 4 * (4 * c_3 * (k * a_24 - b_22) + k * c_4) + \
            3.0/2 * (9 * lmda**2 + 1 - c_2) * (c_3 * (k * b_22 + d_21 - 2 * a_24) - c_4))
        b_31 = 3 / (8 * d_2) * (8 * lmda * (3 * c_3 * (k * b_21 - 2 * a_23) - c_4 * (2 + 3 * k**2)) + \
            (9 * lmda**2 + 1 + 2 * c_2)*(4 * c_3 * (k * a_23 - b_21) + k * c_4 * (4 + k**2)))
        b_32 = 1 / d_2 * (9 * lmda * (3 * c_3 * (k * b_22 + d_21 - 2 * a_24) - c_4) + \
            3.0/8 * (9 * lmda**2 + 1 + 2 * c_2) * (4 * c_3 * (k * a_24 - b_22) + k * c_4))
        d_32 = 3 / (64 * lmda**2) * (4 * c_3 * (a_23 - d_21) + c_4 * (4 + k**2))

        s_1 = (3.0/2 * c_3 * (2 * a_21 * (k**2 - 2) - a_23 * (k**2 + 2) - 2 * k * b_21) - \
            3.0/8 * c_4 * (3 * k**4 - 8 * k**2 + 8)) / (2 * lmda * (lmda * (1 + k**2) - 2 * k))
        s_2 = (3.0/2 * c_3 * (2 * a_22 * (k**2 - 2) + a_24 * (k**2 + 2) + 2 * k * b_22 + 5 * d_21) + \
            3.0/8 * c_4 * (12 - k**2)) / (2 * lmda * (lmda * (1 + k**2) - 2 * k))

        a_1 = -3.0/2 * c_3 * (2 * a_21 + a_23 + 5 * d_21) - 3.0/8 * c_4 * (12 - k**2)
        a_2 = 3.0/2 * c_3 * (a_24 - 2 * a_22) + 9.0/8 * c_4
        l_1 = a_1 + 2 * lmda**2 * s_1
        l_2 = a_2 + 2 * lmda**2 * s_2

        tau_1 = tau_2 = 0
        r = 1 if orbitFamily == self.OrbitFamily.northHalo else 3
        d_r = 2 - r

        Delta = lmda**2 / 4 - c_2  if orbitFamily == self.OrbitFamily.verticalLyapunov else lmda**2 - c_2   
        Ax = np.sqrt(-(l_2 * Az**2 + Delta) / l_1)
        # Raise error if Ax cannot be computed. That is if Az is too big.
        if np.isnan(Ax): 
            maxEnergy = self.getJacobi(np.array( [ distLagrange, 0.0, np.sqrt(-Delta / l_2) * gamma, 0.0, 0.0, 0.0 ] ))
            raise ValueError("Az:", Az, " value too big. Max Az:", np.sqrt(-Delta / l_2), "Max C:", maxEnergy)
            
        w = 1 + s_1 * Ax**2 + s_2 * Az**2
        if orbitFamily == self.OrbitFamily.verticalLyapunov:
            return np.array( [ distLagrange, 0.0, Az * gamma, 0.0, k * Ax * gamma * w, 0.0 ] )
        
        else:
            x = a_21 * Ax**2 + a_22 * Az**2 - Ax * np.cos(tau_1) + (a_23 * Ax**2 - a_24 * Az**2) * np.cos(2 * tau_1) + (a_31 * Ax**3 - a_32 * Ax * Az**2) * np.cos(3 * tau_1)
            z = d_r * Az * np.cos(tau_1 + d_r * np.pi/2 ) + d_r * d_21 * Ax * Az * (np.cos(2 * tau_1 + d_r* np.pi/2) - 3) + d_r * (d_32 * Az * Ax**2 - d_31 * Az**3) * np.cos(3 * tau_1 + d_r * np.pi/2)
            y_dot = k * Ax * np.cos(tau_2) + (b_21 * Ax**2 - b_22 * Az**2) * 2 * np.cos(2 * tau_2) + (b_31 * Ax**3 - b_32 * Ax * Az**2) * 3 * np.cos(3 * tau_2 )

            return np.array( [distLagrange + x * gamma, 0.0, z * gamma, 0.0, y_dot * gamma * w, 0.0] )

    def _approxInitialStatePlanar( self, x, lagrangePoint):
        distanceLagrange =  self.getLagrangePoint(lagrangePoint)[0]
        # TODO: This isonly valid for the Sun-Earth system, generalize
        if (lagrangePoint == self.LagrangePoint.l1):
            s = 2.087
            v = 3.229
        elif (lagrangePoint == self.LagrangePoint.l2):
            s = 2.057
            v = 3.187
        elif (lagrangePoint == self.LagrangePoint.l3):
            s = 1.0
            v = 2.0

        return np.array( [ x, 0, 0, 0, -s * v * (x - distanceLagrange), 0 ] )

    def getPeriodicOrbit( self, energy, lagrangePoint, orbitFamily, initialTime=0.0 ):
        # Check if energy value is valid
        if (energy > self.getJacobi(self.getLagrangePoint(lagrangePoint))):
            raise ValueError("Energy value too high. Max C=", self.getJacobi(self.getLagrangePoint(lagrangePoint)))

        # Try to compute orbit only with predictor/corrector
        try:
            approxInitialState = self.getApproximateInitialState(energy, lagrangePoint, orbitFamily)
            initialState = self.applyCorrectorPredictor(approxInitialState, initialTime, energy, lagrangePoint, orbitFamily)
            return self.runSimulation(initialState, [0.0, var.timeDimensionalToNormalized(400*oneDay)], max_step=var.timeDimensionalToNormalized(10000))

        # Use continuation to compute orbits
        except:
            pass

    def continuateSolution( self, initialState, dL ):
        u = self.massParameter
        x, y, z, dx, dy, dz = initialState[0], initialState[1], initialState[2], initialState[3], initialState[4], initialState[5]
        r1 = np.sqrt((u + x)**2 + y**2 + z**2)
        r2 = np.sqrt((1 - u - x)**2 + y**2 + z**2)

        dHx = - 2 * (x + u) * (1 - u) / r1**3 + 2 * (1 - u - x) * u / r2**3 + 2 * x
        dHy = - 2 *       y * (1 - u) / r1**3 - 2 *           y * u / r2**3 + 2 * y
        dHz = - 2 *       z * (1 - u) / r1**3 - 2 *           z * u / r2**3 
        dHdx = - 2 * dx
        dHdy = - 2 * dy
        dHdz = - 2 * dz

        dH = np.array( [ dHx, dHy, dHz, dHdx, dHdy, dHdz ] )
        return initialState + dL * dH

    def generateDatabase( self, lagrangePoint, orbitFamily, initialParameter=0, initialStateSeed=[], dL=0.1, dLsave=0.1, dLmin=1E-3, totalDeltaL=[], EnergySpan=[], initialTime=0.0, fileName=""):
        # Function to write in file new data
        def writeInFile(fileName, energy, state, time):
            with open(fileName, 'a') as outputFile:
                outputFile.write('{:.16f}'.format(energy) + " " + '{:.16f}'.format(state[0]) + " " + '{:.16f}'.format(state[1]) + " " + '{:.16f}'.format(state[2]) + \
                    " " + '{:.16f}'.format(state[3]) + " " + '{:.16f}'.format(state[4]) + " " + '{:.16f}'.format(state[5]) + " " + '{:.16f}'.format(time) + "\n")

        # Create new file
        if fileName:
            f = open(fileName, "w")
            f.close()

        dLorg = dL # Save for later the original dL

        # Check for valid initial condition
        if not totalDeltaL and not EnergySpan and dLmin == 0:
            raise ValueError("Invalid stopping conditions.")

        # Compute the seed for the continuation process or use the given initialState
        if initialStateSeed:
            initialState = initialStateSeed
        else:
            try:
                print("Computing seed initial state...")
                approxInitialState = self.getApproximateInitialState(initialParameter, lagrangePoint, orbitFamily)
                initialState, terminatingPeriod = self.applyCorrectorPredictor(approxInitialState, initialTime, lagrangePoint, orbitFamily, returnTerminatingPeriod=True)
                print("Completed!")
            except Exception as error:
                print(error)
                raise RuntimeError("There was an error computing the seed state. Try changing the initialParameter value.")
        
        dictDatabase = {self.getJacobi(initialState): (initialState, terminatingPeriod)}
        # Compute continuation, each time dL is too big and the integration throws error dL /= 2 until either dL min is reached or a C boundary.
        # Start with forward integration
        keepForwardContinuing = True
        DeltaLcounter = 0
        totalDeltaLcounter = 0
        print("Starting forward continuation...")
        while keepForwardContinuing:
            try:
                continuedState = self.continuateSolution(initialState, dL)
                initialState, terminatingPeriod = self.applyCorrectorPredictor(continuedState, initialTime, lagrangePoint, orbitFamily, returnTerminatingPeriod=True)
                DeltaLcounter += dL
                totalDeltaLcounter += dL

                # Check if the value should be saved and reset counter
                if DeltaLcounter >= dLsave:
                    stateEnergy = self.getJacobi(initialState)
                    dictDatabase.update({stateEnergy: (initialState, terminatingPeriod)})
                    print("C=",stateEnergy," state saved! Total L=", totalDeltaLcounter)
                    DeltaLcounter = 0

                    if fileName:
                        writeInFile(fileName, stateEnergy, initialState, terminatingPeriod)

            except Exception as error:
                print(error)
                print("An error with dL = ", dL, "was encountered")
                dL /= 2.0
                print("New dL = ", dL)

            # Update the forward loop condition
            if EnergySpan and dLmin != 0 and totalDeltaLcounter != 0:
                keepForwardContinuing = EnergySpan[0] <= self.getJacobi(initialState) <= EnergySpan[1] and abs(dL) > dLmin and abs(totalDeltaLcounter) < totalDeltaL[0]
            elif EnergySpan and dLmin != 0 and totalDeltaLcounter == 0:
                keepForwardContinuing = EnergySpan[0] <= self.getJacobi(initialState) <= EnergySpan[1] and abs(dL) > dLmin
            elif EnergySpan and dLmin == 0 and totalDeltaLcounter != 0:
                keepForwardContinuing = EnergySpan[0] <= self.getJacobi(initialState) <= EnergySpan[1] and abs(totalDeltaLcounter) < totalDeltaL[0]
            elif EnergySpan and dLmin == 0 and totalDeltaLcounter == 0:
                keepForwardContinuing = EnergySpan[0] <= self.getJacobi(initialState) <= EnergySpan[1]
            elif not EnergySpan and dLmin != 0 and totalDeltaLcounter != 0:
                keepForwardContinuing = abs(dL) > dLmin and abs(totalDeltaLcounter) < totalDeltaL[0]
            elif not EnergySpan and dLmin != 0 and totalDeltaLcounter == 0:
                keepForwardContinuing = abs(dL) > dLmin
            elif not EnergySpan and dLmin == 0 and totalDeltaLcounter != 0:
                keepForwardContinuing = abs(totalDeltaLcounter) < totalDeltaL[0]
            else:
                raise ValueError("Error! Invalid stopping conditions")
        
        print("Forward continuation completed!")
        dL = dLorg          # Reset dL
        keepBackwardContinuing = True
        DeltaLcounter = 0   # Reset counters
        totalDeltaLcounter = 0
        print("Starting backward continuation...")
        # Backward continuation
        while keepBackwardContinuing:
            try:
                continuedState = self.continuateSolution(initialState, -dL)
                initialState, terminatingPeriod = self.applyCorrectorPredictor(continuedState, initialTime, lagrangePoint, orbitFamily, returnTerminatingPeriod=True)
                DeltaLcounter += dL
                totalDeltaLcounter += dL

                # Check if the value should be saved and reset counter
                if DeltaLcounter >= dLsave:
                    stateEnergy = self.getJacobi(initialState)
                    dictDatabase.update({stateEnergy: (initialState, terminatingPeriod)})
                    print("C=",stateEnergy," state saved! Total L=", totalDeltaLcounter)
                    DeltaLcounter = 0

                    # Write in file
                    if fileName:
                        writeInFile(fileName, stateEnergy, initialState, terminatingPeriod)

            except Exception as error:
                print(error)
                print("An error with dL = ", dL, "was encountered")
                dL /= 2.0
                print("New dL = ", dL)

            # Update the forward loop condition
            if EnergySpan and dLmin != 0 and totalDeltaLcounter != 0:
                keepBackwardContinuing = EnergySpan[0] <= self.getJacobi(initialState) <= EnergySpan[1] and abs(dL) > dLmin and abs(totalDeltaLcounter) < totalDeltaL[1]
            elif EnergySpan and dLmin != 0 and totalDeltaLcounter == 0:
                keepBackwardContinuing = EnergySpan[0] <= self.getJacobi(initialState) <= EnergySpan[1] and abs(dL) > dLmin
            elif EnergySpan and dLmin == 0 and totalDeltaLcounter != 0:
                keepBackwardContinuing = EnergySpan[0] <= self.getJacobi(initialState) <= EnergySpan[1] and abs(totalDeltaLcounter) < totalDeltaL[1]
            elif EnergySpan and dLmin == 0 and totalDeltaLcounter == 0:
                keepBackwardContinuing = EnergySpan[0] <= self.getJacobi(initialState) <= EnergySpan[1]
            elif not EnergySpan and dLmin != 0 and totalDeltaLcounter != 0:
                keepBackwardContinuing = abs(dL) > dLmin and abs(totalDeltaLcounter) < totalDeltaL[1]
            elif not EnergySpan and dLmin != 0 and totalDeltaLcounter == 0:
                keepBackwardContinuing = abs(dL) > dLmin
            elif not EnergySpan and dLmin == 0 and totalDeltaLcounter != 0:
                keepBackwardContinuing = abs(totalDeltaLcounter) < totalDeltaL[1]
            else:
                raise ValueError("Error! Invalid stopping conditions")

        print("Continuation process finished! Returning database dictionary...")
        return dictDatabase

    def readDatabase( self, fileName="myFile.dat"):
        myDict = dict()
        with open(fileName, 'r') as readFile:
            for line in readFile:
                energy, x, y, z, dx, dy, dz, t = line.split(" ")
                myDict.update({float(energy):(np.array([float(x),float(y),float(z),float(dx),float(dy),float(dz)]), float(t[:-2]))})

        return myDict


testsOn = False
computeFamilies = False

oneDay = 86400
distanceSunEarth = 149597870700 # in m
var = PeriodicOrbit("Sun","Earth", distanceSunEarth)
lagrangePointX = var.getLagrangePoint(var.LagrangePoint.l1)[0]

if testsOn:
    #### Four types of orbit test ####
    lagrangePoint = var.LagrangePoint.l1
    # PLANAR LYAPUNOV
    approxInitialState = var._approxInitialStatePlanar(lagrangePointX-0.0009, lagrangePoint)
    initialState, halfPeriod = var.applyCorrectorPredictor(approxInitialState, 0.0, lagrangePoint, var.OrbitFamily.planarLyapunov, returnTerminatingPeriod=True)
    solData1 = var.runSimulation(initialState, [0.0, 2*halfPeriod], 0.01)
    # HALO NORTH
    approxInitialState = var._approxInitialState3D(1.001, lagrangePoint, var.OrbitFamily.northHalo)
    initialState, halfPeriod = var.applyCorrectorPredictor(approxInitialState, 0.0, lagrangePoint, var.OrbitFamily.northHalo, returnTerminatingPeriod=True)
    solData2 = var.runSimulation(initialState, [0.0, 2*halfPeriod], 0.01)
    # SOUTH NORTH
    approxInitialState = var._approxInitialState3D(1.001, lagrangePoint, var.OrbitFamily.southHalo)
    initialState, halfPeriod = var.applyCorrectorPredictor(approxInitialState, 0.0, lagrangePoint, var.OrbitFamily.southHalo, returnTerminatingPeriod=True)
    solData3 = var.runSimulation(initialState, [0.0, 2*halfPeriod], 0.01)
    # VERTICAL LYAPUNOV
    approxInitialState = var._approxInitialState3D(1, lagrangePoint, var.OrbitFamily.verticalLyapunov)
    initialState, halfPeriod = var.applyCorrectorPredictor(approxInitialState, 0.0, lagrangePoint, var.OrbitFamily.verticalLyapunov, returnTerminatingPeriod=True)
    solData4 = var.runSimulation(initialState, [0.0, 4*halfPeriod], 0.01)

    # Plot solutions
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(solData1.y[0], solData1.y[1], solData1.y[2],'blue')
    ax.plot3D(solData2.y[0], solData2.y[1], solData2.y[2],'blue')
    ax.plot3D(solData3.y[0], solData3.y[1], solData3.y[2],'blue')
    ax.plot3D(solData4.y[0], solData4.y[1], solData4.y[2],'blue')
    ax.scatter(lagrangePointX, 0, 0,  c = 'r', marker='x')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

elif computeFamilies :

    fileName1 = "planarLyapunov.dat"

    # Continuation tests
    # lagrangePoint = var.LagrangePoint.l1
    # var.generateDatabase(lagrangePoint, var.OrbitFamily.planarLyapunov, var.getLagrangePoint(lagrangePoint)[0]-0.0001, 
    #                     dL=0.001, dLsave=0.01, dLmin=1E-4, totalDeltaL=[2,2], fileName=fileName)

    # Read file
    myDict1 = var.readDatabase(fileName1)

    # Plot from solution dictionary
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for key in myDict1:
        initialState, terminatingPeriod = myDict1[key]
        solData = var.runSimulation(initialState, [0.0, 4*terminatingPeriod], max_step=0.01)
        ax.plot3D(solData.y[0], solData.y[1], solData.y[2],'blue')
    ax.scatter(lagrangePointX, 0, 0,  c = 'r', marker='x')
    ax.scatter(1-var.massParameter,0,0,'yellow')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

else:
    # Plot from solution dictionary
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # PLANAR
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    fileName = "planarLyapunov.dat"
    myDict = var.readDatabase(fileName)
    # Plot data
    for i, key in enumerate(myDict):
        initialState, terminatingPeriod = myDict[key]
        solData = var.runSimulation(initialState, [0.0, 4*terminatingPeriod], max_step=0.01)
        alpha = i * (1-0.2) / len(myDict)
        ax1.plot3D(solData.y[0], solData.y[1], solData.y[2],'blue',alpha=alpha)

    # HALO NORTH
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    fileName = "northHaloL1.dat"
    myDict = var.readDatabase(fileName)
    # Plot data
    for i, key in enumerate(myDict):
        initialState, terminatingPeriod = myDict[key]
        solData = var.runSimulation(initialState, [0.0, 4*terminatingPeriod], max_step=0.01)
        alpha = i * (0.8-0.2) / len(myDict)
        ax2.plot3D(solData.y[0], solData.y[1], solData.y[2],'red',alpha=alpha)

    # SOUTH NORTH
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    fileName = "southHaloL1.dat"
    myDict = var.readDatabase(fileName)
    # Plot data
    for i, key in enumerate(myDict):
        initialState, terminatingPeriod = myDict[key]
        solData = var.runSimulation(initialState, [0.0, 4*terminatingPeriod], max_step=0.01)
        alpha = i * (0.8-0.2) / len(myDict)
        ax3.plot3D(solData.y[0], solData.y[1], solData.y[2],'black',alpha=alpha)

    # VERTICAL
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    fileName = "verticalLyapunov.dat"
    myDict = var.readDatabase(fileName)
    # Plot data
    for i, key in enumerate(myDict):
        initialState, terminatingPeriod = myDict[key]
        solData = var.runSimulation(initialState, [0.0, 4*terminatingPeriod], max_step=0.01)
        alpha = i * (1-0.2) / len(myDict)
        ax4.plot3D(solData.y[0], solData.y[1], solData.y[2],'green',alpha=alpha)

    ax1.scatter(lagrangePointX, 0, 0,  c = 'r', marker='x')
    ax2.scatter(lagrangePointX, 0, 0,  c = 'r', marker='x')
    ax3.scatter(lagrangePointX, 0, 0,  c = 'r', marker='x')
    ax4.scatter(lagrangePointX, 0, 0,  c = 'r', marker='x')
    ax1.set_title('Planar Lyapunov')
    ax2.set_title('North Halo')
    ax3.set_title('South Halo')
    ax4.set_title('Vertical Lyapunov')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    plt.show()



