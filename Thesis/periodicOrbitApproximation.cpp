#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <Eigen/Core>
#include <math.h>

#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/SimulationSetup/PropagationSetup/propagationCR3BPFullProblem.h"
#include "Tudat/Mathematics/RootFinders/rootFinder.h"
#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"
#include "Tudat/Mathematics/BasicMathematics/functionProxy.h"

namespace tudat
{

namespace circular_restricted_three_body_problem
{

class PeriodicOrbitApproximation
{
public:

    //! Lagrange libration points.
    enum LagrangeLibrationPoints { l1, l2, l3 };
    //! Periodic orbit families.
    enum PeriodicOrbitFamilies { planarLyapunov };

    //! Default constructor.
    /*!
     * Default constructor.
     * \param aMassParameter Dimensionless mass parameter of the smaller of the massive bodies in the CRTBP.
     * \param aRootFinder Shared pointer to the rootfinder which is used for finding the approximate initial
     * state for the periodic orbit selected.
     */
    PeriodicOrbitApproximation( const double massParameter,
                    const tudat::root_finders::RootFinderPointer aRootFinder )
        : massParameter( massParameter ),
          rootFinder( aRootFinder )
    { }

    //! Solve the approximate initial conditions for a periodic orbit problem.
    /*!
     * Returns the approximate initial conditions for a periodic orbit problem for a
     * selected settings.
     * \param energyLevel Jacobi integral value for the periodic orbit.
     * \param lagrangeLibrationPoint Equilibrium point to create periodic orbit around.
     * \param orbitFamily Type of periodic orbit: planar or vertical Lyapunov or Halo.
     * \param initialGuess Initial guess for root solving, by default eq. point selected.
     * \param verbose Whether to print extra information or not. Default false.
     * \return Solved initial approximated conditions for periodic orbit.
     */
    void approxInitialStatePeriodicOrbit( double energyLevel, LagrangeLibrationPoints lagrangeLibrationPoint, PeriodicOrbitFamilies orbitFamily,
                                          bool verbose=false, double initialGuess=0.0)
    {
        this->energyLevel = energyLevel;
        this->firstOrderConstants = computeConstants(lagrangeLibrationPoint);
        double lagrangeEnergy = getJacobiAtLagrangePoint(lagrangeLibrationPoint);

        // If verbose, print extra information.
        if (verbose == true) {
            std::cout << "Distance to eq. point: " << distLagrangePoint << std::endl;
            std::cout << "C at eq. point: " << lagrangeEnergy << std::endl;
            std::cout << "C selected    : " << energyLevel << std::endl;
        }

        // Check the selected energyLevel is valid.
        if (energyLevel > lagrangeEnergy)
            throw std::invalid_argument( "Energy level cannot be larger than equilibirum point energy level!" );

        // If inital guess not provided use equilibrum point distance
        if (initialGuess == 0.0)
            initialGuess = distLagrangePoint;

        // Solve the root finding problem according to the selected family
        switch( orbitFamily )
        {
        case planarLyapunov:
            tudat::basic_mathematics::UnivariateProxyPointer rootFunction = std::make_shared< tudat::basic_mathematics::UnivariateProxy >(
                        std::bind( &PeriodicOrbitApproximation::computePlanarLyapunovInitialStateFunction, this, std::placeholders::_1 ) );

            rootFunction->addBinding( -1, std::bind( &PeriodicOrbitApproximation::
                    computePlanarLyapunovInitialStateFunctionDerivative, this, std::placeholders::_1 ) );

            double solRoot = rootFinder->execute( rootFunction, initialGuess );
            periodicInitialState_ << solRoot, 0., 0. , 0., -firstOrderConstants.at("s") * firstOrderConstants.at("v") * (solRoot - distLagrangePoint), 0.;

            if (verbose == true) {
            std::cout << "Approximated inital state: " << "( " << periodicInitialState_[0] << ", " << periodicInitialState_[1] <<
                         ", " << periodicInitialState_[2] << ", " << periodicInitialState_[3] << ", " << periodicInitialState_[4] <<
                         ", " << periodicInitialState_[5] << " )" << std::endl;
            }
        }
    }

    //! Returns the approximated initial state for the periodic orbit.
    Eigen::Vector6d getApproxInitialStatePeriodicOrbit( )
    {
        // Check if approxInitialStatePeriodicOrbit has been called before
        if (distLagrangePoint == 0)
            throw std::invalid_argument( "pointLagrange argument not valid" );

        return periodicInitialState_;
    }

    //! Get Jacobi energy at libration point
    /*!
     * Returns jacobi constant value at equilibrium point.
     * \param lagrangeLibrationPoint Lagrange point selected for getting energy.
     * \return Jacobi constant value.
     */
    double getJacobiAtLagrangePoint(LagrangeLibrationPoints lagrangeLibrationPoint)
    {
        Eigen::Vector6d stateLagrangePoint;
        Eigen::Vector3d posLagrangePoint = computeLagrangePoint(lagrangeLibrationPoint);
        stateLagrangePoint << posLagrangePoint[0], posLagrangePoint[1], posLagrangePoint[2], 0, 0, 0;

        return tudat::gravitation::computeJacobiEnergy(massParameter, stateLagrangePoint);
    }


private:
    const double massParameter;
    double distLagrangePoint = 0, energyLevel = 0;;
    const tudat::root_finders::RootFinderPointer rootFinder;
    std::map< string, double> firstOrderConstants;
    Eigen::Vector6d periodicInitialState_;

    //! Get the constants for inital state equation according to the equilibrium point.
    /*!
     * Returns constants required for the linearized initial state equation.
     * \param numberLagrange Lagrange point selected for getting the approximate state.
     * \return Map with the computed constants required for the equilibrium point selected.
     */
    std::map< string, double> computeConstants( LagrangeLibrationPoints numberLagrange )
    {
        this->distLagrangePoint = computeLagrangePoint(numberLagrange)[0];

        std::map< string, double> constants;
        switch(numberLagrange) {
           case l1 :
                constants.insert({"s", 2.087});
                constants.insert({"v", 3.229});
              break;
           case l2 :
              break;
           case l3 :
              break;
           default :
              throw std::invalid_argument( "pointLagrange argument not valid" );
        }
        return constants;
    }

    //! Get location of Lagrange libration point.
    /*!
     * Returns the position vector in Cartesian elements of a Lagrange libration point.
     * \param numberLagrange Lagrange point selected for getting the approximate state
     * \return Cartesian position elements of Lagrange libration point.
     */
    Eigen::Vector3d computeLagrangePoint( LagrangeLibrationPoints numberLagrange )
    {
        tudat::circular_restricted_three_body_problem::LibrationPoint librationPoint =
                tudat::circular_restricted_three_body_problem::LibrationPoint(massParameter, rootFinder);
        switch (numberLagrange) {
            case l1 :
                librationPoint.computeLocationOfLibrationPoint( librationPoint.l1 );
                break;
            case l2 :
                librationPoint.computeLocationOfLibrationPoint( librationPoint.l2 );
                break;
            case l3 :
                librationPoint.computeLocationOfLibrationPoint( librationPoint.l3 );
                break;
            default :
               throw std::invalid_argument( "Langrange point not valid. Only L1,2,3." );
            }

        Eigen::Vector3d pointLagrange = librationPoint.getLocationOfLagrangeLibrationPoint();
        return pointLagrange;
    }

    //! Compute linearized equation for colinear periodic orbits for a given energy level.
    /*!
     * Computes the linearized initial conditions for colinear periodic orbits where
     * y_dot is substituted by a f(H,x) (F. Wakker, 2015).
     * \param xEstimate Estimate of x-location for initial conditions.
     * \return Value of the equation.
     */
    double computePlanarLyapunovInitialStateFunction( const double xEstimate )
    {
        return sqrt(2 * (massParameter / (1 - massParameter - xEstimate) + (1 - massParameter) / (massParameter + xEstimate) + pow(xEstimate, 2) / 2) - energyLevel) +
                firstOrderConstants.at("s") * firstOrderConstants.at("v") * (xEstimate - distLagrangePoint);
    }

    //! Compute first derivative of linearized equation for colinear periodic orbits for a given energy level.
    /*!
     * Computes the first derivative of the linearized initial conditions for colinear periodic orbits where
     * y_dot is substituted by a f(H,x) (F. Wakker, 2015).
     * \param xEstimate Estimate of x-location for initial conditions.
     * \return Value of the first derivative equation.
     */
    double computePlanarLyapunovInitialStateFunctionDerivative( const double xEstimate )
    {
        return (xEstimate - (1 - massParameter) / pow(xEstimate + massParameter, 2) + massParameter / pow(1 - massParameter - xEstimate, 2)) /
                sqrt(2 * (massParameter / (1 - massParameter - xEstimate) + (1 - massParameter) / (massParameter + xEstimate) + pow(xEstimate, 2) / 2) - energyLevel) +
                firstOrderConstants.at("s") * firstOrderConstants.at("v");
    }
};

} // namespace circular_restricted_three_body_problem

} // namespace tudat
