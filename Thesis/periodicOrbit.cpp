#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include <Eigen/Core>

#include <Tudat/Astrodynamics/BasicAstrodynamics/physicalConstants.h>
#include <Tudat/Basics/testMacros.h>
#include <Tudat/Mathematics/BasicMathematics/mathematicalConstants.h>
#include "Tudat/Astrodynamics/BasicAstrodynamics/unitConversions.h"
#include <Tudat/Astrodynamics/BasicAstrodynamics/orbitalElementConversions.h>
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/unitConversionsCircularRestrictedThreeBodyProblem.h"
#include "Tudat/SimulationSetup/PropagationSetup/propagationCR3BPFullProblem.h"

#include "Thesis/applicationOutput.h"
#include <Tudat/InputOutput/basicInputOutput.h>
#include "Tudat/Mathematics/RootFinders/rootFinder.h"
#include <Tudat/SimulationSetup/tudatSimulationHeader.h>
#include "Tudat/Astrodynamics/Ephemerides/approximatePlanetPositions.h"
#include "Tudat/Astrodynamics/Gravitation/unitConversionsCircularRestrictedThreeBodyProblem.h"
#include "Tudat/Astrodynamics/OrbitDetermination/EstimatableParameters/estimatableParameter.h"

#include "periodicOrbitApproximation.cpp"

namespace tudat
{

namespace circular_restricted_three_body_problem
{

class PeriodicOrbit
{


public:
    //! Lagrange libration points.
    enum LagrangeLibrationPoints { l1, l2, l3 };
    //! Periodic orbit families.
    enum PeriodicOrbitFamilies { planarLyapunov };

    PeriodicOrbit( const string bodyPrimary, const string bodySecondary, const double distanceBodies ) :
        bodyPrimary(bodyPrimary),
        bodySecondary(bodySecondary),
        distanceBodies(distanceBodies)
    {
        // Load Spice kernels.
        spice_interface::loadStandardSpiceKernels( );

        // Define final time for the propagation.
        this->gravitationalParameterPrimary = createGravityFieldModel(
                    simulation_setup::getDefaultGravityFieldSettings(
                        bodyPrimary, TUDAT_NAN, TUDAT_NAN ), bodyPrimary )->getGravitationalParameter( );
        this->gravitationalParameterSecondary = createGravityFieldModel(
                    simulation_setup::getDefaultGravityFieldSettings(
                        bodySecondary, TUDAT_NAN, TUDAT_NAN ), bodySecondary )->getGravitationalParameter( );

        // Compute mass parameter.
        this->massParameter = circular_restricted_three_body_problem::computeMassParameter(
                    gravitationalParameterPrimary, gravitationalParameterSecondary );

        this->bodyMap = propagators::setupBodyMapCR3BP(distanceBodies, bodyPrimary, bodySecondary, "Spacecraft" );

        // Define propagator settings variables.
        this->bodiesToPropagate.push_back( "Spacecraft" );
        this->centralBodies.push_back( bodyPrimary ); // SSB for Solar System Barycenter

        // CR3BP problem acceleration map.
        this->accelerationModelMap = propagators::setupAccelerationMapCR3BP(
                    bodyPrimary, bodySecondary, bodiesToPropagate.at( 0 ), centralBodies.at( 0 ), bodyMap );
    }


    void computePeriodicOrbit(double energyLevel, LagrangeLibrationPoints lagrangeLibrationPoint, PeriodicOrbitFamilies orbitFamily,
                              bool verbose=false, double initialGuess=0.0)
    {
        Eigen::Vector6d initialState = getApproximateInitalState(energyLevel, lagrangeLibrationPoint, orbitFamily, verbose, initialGuess);
        std::cout << "initialState: " << initialState << std::endl;
        propagators::SingleArcVariationalEquationsSolver< > variationalEquationsSimulator = setUpProgragation(0.0);

        //double xDotError = 100;
        std::map< double, Eigen::VectorXd> cr3bpPropagation;
        for ( int i = 0; i < 3; i++)
        {   
        variationalEquationsSimulator.integrateVariationalAndDynamicalEquations(initialState, true);

        cr3bpPropagation = variationalEquationsSimulator.getDynamicsSimulator( )->getEquationsOfMotionNumericalSolution( );
        std::map< double, Eigen::MatrixXd > stateTransitionResult =
                variationalEquationsSimulator.getNumericalVariationalEquationsSolution( ).at( 0 );
        std::map< double, Eigen::MatrixXd > sensitivityResult =
                variationalEquationsSimulator.getNumericalVariationalEquationsSolution( ).at( 1 );

        Eigen::Vector6d lastCr3bpStateNormalized = convertCartesianToCorotatingNormalizedCoordinates(
                    gravitationalParameterPrimary, gravitationalParameterSecondary, distanceBodies, (--cr3bpPropagation.end( ))->second, (--cr3bpPropagation.end( ))->first);

        //std::cout << "lastCr3bpStateNormalized: " << lastCr3bpStateNormalized << std::endl;
        double xDotError = lastCr3bpStateNormalized[3];
        auto StateTransitionCartesian = (--stateTransitionResult.end())->second;

        double initalTimeCR3BP = convertDimensionalTimeToDimensionlessTime( cr3bpPropagation.begin( )->first,
                                 gravitationalParameterPrimary, gravitationalParameterSecondary, distanceBodies);
        double finalTime = (--cr3bpPropagation.end( ))->first;

        auto StateTransition = CartesianStateTransitionMatrixtoCR3BP(StateTransitionCartesian,
                                 gravitationalParameterPrimary, gravitationalParameterSecondary, distanceBodies, initalTimeCR3BP, finalTime);

        Eigen::Vector6d correctionCR3BP;
        correctionCR3BP << 0, 0, 0, 0, -xDotError / StateTransition(3,4), 0;

        Eigen::Vector6d correctionCartesian = convertCorotatingNormalizedToCartesianCoordinates(
                    gravitationalParameterPrimary, gravitationalParameterSecondary, distanceBodies, correctionCR3BP, initalTimeCR3BP);


        std::cout << "xDotError: " << xDotError << std::endl;

        //std::cout << "correctionCartesian: " << correctionCartesian << std::endl;

        initialState += correctionCartesian;
        //std::cout << "corrected: " << initialState << std::endl;
        }

        // Transform to normalized corotating coordinates
        std::map< double, Eigen::Vector6d > cr3bpNormalisedCoRotatingFrame;
        for( std::map< double, Eigen::VectorXd >::iterator itr = cr3bpPropagation.begin( );
            itr != cr3bpPropagation.end( ); itr++ ){
            cr3bpNormalisedCoRotatingFrame[ itr->first ] = convertCartesianToCorotatingNormalizedCoordinates(
                gravitationalParameterPrimary, gravitationalParameterSecondary, distanceBodies, itr->second, itr->first);
            }

        std::cout << "Writing files..." << std::endl;
        // Write normalized corotating frame state
        input_output::writeDataMapToTextFile( cr3bpNormalisedCoRotatingFrame,
                                              "CR3BPnormalisedCoRotatingFrame.dat",
                                              tudat_applications::getOutputPath( ),
                                              "",
                                              std::numeric_limits< double >::digits10,
                                              std::numeric_limits< double >::digits10,
                                              "," );

    };

private:
    const string bodyPrimary, bodySecondary;
    double distanceBodies, massParameter, gravitationalParameterPrimary, gravitationalParameterSecondary;
    std::vector< std::string > bodiesToPropagate, centralBodies;
    basic_astrodynamics::AccelerationMap accelerationModelMap;
    simulation_setup::NamedBodyMap bodyMap;

    Eigen::Vector6d getApproximateInitalState(double energyLevel, LagrangeLibrationPoints lagrangeLibrationPoint, PeriodicOrbitFamilies orbitFamily,
                                              bool verbose=false, double initialGuess=0.0)
    {
        // Define root solver
        std::shared_ptr< root_finders::NewtonRaphsonCore< double > > rootFinder = std::make_shared< root_finders::NewtonRaphsonCore< double >>(1e-12, 100);

        // Get approximate initial condition
        PeriodicOrbitApproximation approxInitialState = PeriodicOrbitApproximation(massParameter, rootFinder);

        // Switches
        PeriodicOrbitApproximation::LagrangeLibrationPoints lagrangePointInitalState;
        switch (lagrangeLibrationPoint) {
            case l1 :
                lagrangePointInitalState = approxInitialState.l1;
                break;
            case l2 :
                lagrangePointInitalState = approxInitialState.l2;
                break;
            case l3 :
                lagrangePointInitalState = approxInitialState.l3;
                break;
        }
        PeriodicOrbitApproximation::PeriodicOrbitFamilies orbitFamilyInitialState;
        switch (orbitFamily) {
            case planarLyapunov :
                orbitFamilyInitialState = approxInitialState.planarLyapunov;
                break;
        }

        approxInitialState.approxInitialStatePeriodicOrbit(energyLevel, lagrangePointInitalState, orbitFamilyInitialState, verbose, initialGuess);
        Eigen::Vector6d initialStateNormalized = approxInitialState.getApproxInitialStatePeriodicOrbit();

        Eigen::Vector6d initialState = circular_restricted_three_body_problem::convertCorotatingNormalizedToCartesianCoordinates(
                    gravitationalParameterPrimary, gravitationalParameterSecondary, distanceBodies, initialStateNormalized, 0);

        return initialState;
    }

    propagators::SingleArcVariationalEquationsSolver< > setUpProgragation( double initialTime )
    {
        // Termination condition
        auto TerminationCondition = [&](const double time, std::function< Eigen::VectorXd() > stateSpacecraft)
        {
            auto StateNormalized = circular_restricted_three_body_problem::convertCartesianToCorotatingNormalizedCoordinates(
                        gravitationalParameterPrimary, gravitationalParameterSecondary, distanceBodies, stateSpacecraft(), time);
            if (StateNormalized[1] < 0.0)
                return true;
            return false;
        };

        std::shared_ptr< propagators::SingleDependentVariableSaveSettings > terminationDependentVariable =
              std::make_shared< propagators::SingleDependentVariableSaveSettings >(
                  propagators::position_CR3BP, "Spacecraft", "Sun", 1 );

        // Define root solver
        std::shared_ptr< root_finders::RootFinderSettings > rootFinderSettings =
                std::make_shared<root_finders::RootFinderSettings>(root_finders::secant_root_finder, 0.0000001, 100);

        std::shared_ptr<propagators::PropagationDependentVariableTerminationSettings> terminationDependentSettings =
                std::make_shared<propagators::PropagationDependentVariableTerminationSettings>(
                    terminationDependentVariable, 0.0, true, true, rootFinderSettings );

        std::function< Eigen::VectorXd( ) > spacecraftStateFunction =
                std::bind( &simulation_setup::Body::getState, bodyMap.at( "Spacecraft" ) );

        std::shared_ptr< propagators::PropagationTerminationSettings > terminationCustomSettings =
                std::make_shared< propagators::PropagationCustomTerminationSettings >(
                    std::bind( TerminationCondition, std::placeholders::_1, spacecraftStateFunction ) );

        std::vector< std::shared_ptr< propagators::PropagationTerminationSettings > > terminationSettingsList;
        terminationSettingsList.push_back( terminationCustomSettings );
        terminationSettingsList.push_back( terminationDependentSettings );

        const double fixedStepSize = 10000.0; //100000.0;
        std::shared_ptr< numerical_integrators::IntegratorSettings< > > integratorSettings =
                std::make_shared < numerical_integrators::IntegratorSettings < > >
                ( numerical_integrators::rungeKutta4, initialTime, fixedStepSize);

        // Dummy inital state
        Eigen::Vector6d initialState;

        // Define settings for propagation of translational dynamics.
        std::shared_ptr< propagators::TranslationalStatePropagatorSettings< double > > propagatorSettings =
                std::make_shared< propagators::TranslationalStatePropagatorSettings< double > >(
                    centralBodies, accelerationModelMap, bodiesToPropagate, initialState, terminationCustomSettings);

        // Define list of parameters to estimate.
        std::vector< std::shared_ptr<estimatable_parameters::EstimatableParameterSettings > > parameterNames;
        parameterNames.push_back( std::make_shared< estimatable_parameters::InitialTranslationalStateEstimatableParameterSettings< double > >(
                                      "Spacecraft", initialState, "Earth" ) );
        // Create parameters.
        std::shared_ptr< estimatable_parameters::EstimatableParameterSet< double > > parametersToEstimate =
                createParametersToEstimate( parameterNames, bodyMap );

        // Print identifiers and indices of parameters to terminal.
        printEstimatableParameterEntries( parametersToEstimate );

        // Create simulation object and propagate dynamics.
        propagators::SingleArcVariationalEquationsSolver< > variationalEquationsSimulator(
                    bodyMap, integratorSettings, propagatorSettings, parametersToEstimate, false,
                    std::shared_ptr< numerical_integrators::IntegratorSettings< double > >( ), false, false );

        return variationalEquationsSimulator;
    }

};
}
}

int main(){
    using namespace tudat;
    using namespace tudat::input_output;
    using namespace tudat::simulation_setup;
    using namespace tudat::propagators;
    using namespace tudat::circular_restricted_three_body_problem;
    using namespace tudat::estimatable_parameters;

    PeriodicOrbit myOrbit = PeriodicOrbit("Sun", "Earth", 147.83e9);
    myOrbit.computePeriodicOrbit(3.000890, myOrbit.l1, myOrbit.planarLyapunov);
    return 0;
}
