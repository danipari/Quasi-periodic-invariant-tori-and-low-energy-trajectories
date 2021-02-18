#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include <Eigen/Core>
#include <math.h>

#include <Tudat/Astrodynamics/BasicAstrodynamics/physicalConstants.h>
#include <Tudat/Basics/testMacros.h>
#include <Tudat/Mathematics/BasicMathematics/mathematicalConstants.h>
#include "Tudat/Astrodynamics/BasicAstrodynamics/unitConversions.h"
#include <Tudat/Astrodynamics/BasicAstrodynamics/orbitalElementConversions.h>
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/Gravitation/unitConversionsCircularRestrictedThreeBodyProblem.h"
#include "Tudat/SimulationSetup/PropagationSetup/propagationCR3BPFullProblem.h"

#include "Thesis/applicationOutput.h"
#include <Tudat/InputOutput/basicInputOutput.h>
#include "Tudat/Mathematics/RootFinders/rootFinder.h"
#include "Tudat/Mathematics/RootFinders/newtonRaphson.h"
#include <Tudat/SimulationSetup/tudatSimulationHeader.h>
#include "Tudat/Astrodynamics/Ephemerides/approximatePlanetPositions.h"
#include "Tudat/Astrodynamics/Gravitation/unitConversionsCircularRestrictedThreeBodyProblem.h"
#include "Tudat/Astrodynamics/OrbitDetermination/EstimatableParameters/estimatableParameter.h"
#include "Tudat/Mathematics/BasicMathematics/functionProxy.h"


class PeriodicOrbitApproximation
{
public:

    //! Lagrange libration points.
    enum LagrangeLibrationPoints { l1, l2, l3 };

    enum PeriodicOrbitFamilies { planarLyapunov };

    //! Default constructor.
    PeriodicOrbitApproximation( const double massParameter,
                    const tudat::root_finders::RootFinderPointer aRootFinder )
        : massParameter( massParameter ),
          rootFinder( aRootFinder )
    { }

    Eigen::Vector6d getLocationOfLagrangeLibrationPoint( )
    {
        if (distLagrangePoint == 0)
            throw std::invalid_argument( "pointLagrange argument not valid" );

        return periodicInitialState_;
    }

    void approxInitialStatePeriodicOrbit( LagrangeLibrationPoints lagrangeLibrationPoint, PeriodicOrbitFamilies orbitFamily )
    {
        switch( lagrangeLibrationPoint )
        {
        case l1:
        {
            firstOrderConstants = computeConstants(1);
            break;
        }
        case l2:
        {
            firstOrderConstants = computeConstants(2);
            break;
        }
        case l3:
        {
            firstOrderConstants = computeConstants(3);
            break;
        }
        }

        switch( orbitFamily )
        {
        case planarLyapunov:
            tudat::basic_mathematics::UnivariateProxyPointer rootFunction = std::make_shared< tudat::basic_mathematics::UnivariateProxy >(
                        std::bind( &PeriodicOrbitApproximation::computePlanarLyapunovInitialStateFunction, this, std::placeholders::_1 ) );

            rootFunction->addBinding( -1, std::bind( &PeriodicOrbitApproximation::
                    computePlanarLyapunovInitialStateFunctionDerivative, this, std::placeholders::_1 ) );

            double solRoot = rootFinder->execute( rootFunction, -0.001 );
            periodicInitialState_ << solRoot, 0.0, 0.0 , -solRoot * firstOrderConstants.at("w0"), 0.0, 0.0;
        }
    }

private:
    const double massParameter;
    double distLagrangePoint = 0;
    const tudat::root_finders::RootFinderPointer rootFinder;
    Eigen::Vector6d periodicInitialState_;
    std::map< string, double> firstOrderConstants;

    std::map< string, double> computeConstants( int numberLagrange)
    {
        std::map< string, double> constants;
        double dPrimary = (1 - massParameter) - computeDistaceToPrimary( numberLagrange );
        double c2 = 0;
        switch(numberLagrange) {
           case 1 :
              c2 = (massParameter + (1 - massParameter) * pow(dPrimary, 3) / pow((1 - dPrimary), 3)) / pow(dPrimary, 3);
              break;
           case 2 :
              c2 = (massParameter + (1 - massParameter) * pow(dPrimary, 3) / pow((1 + dPrimary), 3)) / pow(dPrimary, 3);
              break;
           case 3 :
              c2 = (1 - massParameter + massParameter * pow(dPrimary, 3) / pow((1 + dPrimary), 3)) / pow(dPrimary, 3);
              break;
           default :
              throw std::invalid_argument( "pointLagrange argument not valid" );
        }

        constants.insert({"w0", sqrt((2 - c2 + sqrt(9 * c2* c2 - 8 * c2)) / 2)});
        constants.insert({"v0", sqrt(c2)});
        constants.insert({"k0", -(constants.at("w0") * constants.at("w0") + 1 + 2 * c2) / (2 * constants.at("w0"))});
        std::cout << dPrimary << " " << c2 << " " << constants.at("w0") << std::endl;
        return constants;
    }

    Eigen::Vector3d computeLagrangePoint( int numberLagrange )
    {
        tudat::circular_restricted_three_body_problem::LibrationPoint librationPoint =
                tudat::circular_restricted_three_body_problem::LibrationPoint(massParameter, rootFinder);
        switch (numberLagrange) {
            case 1 :
                librationPoint.computeLocationOfLibrationPoint( librationPoint.l1 );
                break;
            case 2 :
                librationPoint.computeLocationOfLibrationPoint( librationPoint.l2 );
                break;
            case 3 :
                librationPoint.computeLocationOfLibrationPoint( librationPoint.l3 );
                break;
            default :
               throw std::invalid_argument( "Langrange point not valid. Only L1,2,3." );
            }

        Eigen::Vector3d pointLagrange = librationPoint.getLocationOfLagrangeLibrationPoint();
        return pointLagrange;
    }

    double computeDistaceToPrimary( int numberLagrange )
    {
        this->distLagrangePoint = computeLagrangePoint(numberLagrange)[0];
        return distLagrangePoint;
    }

    double computePlanarLyapunovInitialStateFunction( double alpha )
    {
        return (alpha + distLagrangePoint) * (alpha + distLagrangePoint) + 2 * (1 - massParameter) / (massParameter + (alpha + distLagrangePoint)) +
                2 * massParameter / (1 - massParameter - (alpha + distLagrangePoint)) + pow((firstOrderConstants.at("w0") * (alpha + distLagrangePoint)), 2) - 6;
    }

    double computePlanarLyapunovInitialStateFunctionDerivative( double alpha )
    {
        return 2 * (alpha + distLagrangePoint) - 2 * (1 - massParameter) / pow((massParameter + (alpha + distLagrangePoint)), 2) +
                2 * massParameter / pow((1 - massParameter - (alpha + distLagrangePoint)), 2) + 2 * pow(firstOrderConstants.at("w0"), 2) * (alpha + distLagrangePoint);
    }
};

int main( )
{
    using namespace tudat;
    double massParameter = 3.00348e-06;

    std::cout << "Computing Lagrange point..." << std::endl;
    // Create LibrationPoint  object
    std::shared_ptr< root_finders::NewtonRaphsonCore< double > > rootFinder =
            std::make_shared< root_finders::NewtonRaphsonCore< double >>(1e-12, 100);
    PeriodicOrbitApproximation initialState = PeriodicOrbitApproximation(massParameter, rootFinder);
    // Select point
    initialState.approxInitialStatePeriodicOrbit(initialState.l1, initialState.planarLyapunov);
    // Retrieve point
    Eigen::Vector6d pointsLagrange = initialState.getLocationOfLagrangeLibrationPoint();
    // Print result
    std::cout << pointsLagrange << std::endl;

    // Final statement.
    // The exit code EXIT_SUCCESS indicates that the program was successfully executed.
    return EXIT_SUCCESS;
}
