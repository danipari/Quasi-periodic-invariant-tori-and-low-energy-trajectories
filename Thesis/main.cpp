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

bool TerminationCondition(double time, std::function< Eigen::VectorXd() > stateSpacecraft)
{
    if ((time > 10) && (abs(stateSpacecraft()[0]) < 1E-6) && (stateSpacecraft()[5] > 0))
        return true;
    return false;
};

int main( )
{
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////            USING STATEMENTS              //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    using namespace tudat;
    using namespace tudat::input_output;
    using namespace tudat::simulation_setup;
    using namespace tudat::propagators;
    using namespace tudat::circular_restricted_three_body_problem;
    using namespace tudat::estimatable_parameters;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////     CREATE ENVIRONMENT AND VEHICLE       //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Load Spice kernels.
    spice_interface::loadStandardSpiceKernels( );

    // Define primaries
    string primary = "Sun";
    string secondary = "Earth";

    // // Set simulation time settings.
    double initialTime = 0.0;

    // Global characteristics of the problem
    // (only if no normalized units are going to be used)
    double distanceBodies = 147.83e9;

    // Initialise the spacecraft state (B. Taylor, D. (1981). Horseshoe periodic orbits in the restricted problem of three bodies
    // for a sun-Jupiter mass ratio. Astronomy and Astrophysics. 103. 288-294.)
    Eigen::Vector6d initialState = Eigen::Vector6d::Zero();
    initialState[0] = - 140.83e9;
    initialState[4] =  -1.29e4;

    // Create body map.
    std::vector < std::string > bodiesCR3BP;
    bodiesCR3BP.push_back( primary );
    bodiesCR3BP.push_back( secondary );

    // Define propagator settings variables.
    std::vector< std::string > bodiesToPropagate;
    std::vector< std::string > centralBodies;
    bodiesToPropagate.push_back( "Spacecraft" );
    centralBodies.push_back( primary );  // SSB for Solar System Barycenter

    NamedBodyMap bodyMap = setupBodyMapCR3BP(distanceBodies, primary, secondary, "Spacecraft" );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////            CREATE ACCELERATIONS          //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Define final time for the propagation.
    double gravitationalParameterPrimary = createGravityFieldModel(
                getDefaultGravityFieldSettings(
                    primary, TUDAT_NAN, TUDAT_NAN ), primary )->getGravitationalParameter( );
    double gravitationalParameterSecondary = createGravityFieldModel(
                getDefaultGravityFieldSettings(
                    secondary, TUDAT_NAN, TUDAT_NAN ), secondary )->getGravitationalParameter( );
    double finalTime = convertDimensionlessTimeToDimensionalTime(
                1 * ( 2.0 * mathematical_constants::PI ), gravitationalParameterPrimary, gravitationalParameterSecondary, distanceBodies);

    // Compute mass parameter.
    double massParameter = circular_restricted_three_body_problem::computeMassParameter(
                gravitationalParameterPrimary, gravitationalParameterSecondary );
    std::cout << "Mass Parameter: " << massParameter << std::endl;

    // CR3BP problem acceleration map.
    basic_astrodynamics::AccelerationMap accelerationModelMap = propagators::setupAccelerationMapCR3BP(
            primary, secondary, bodiesToPropagate.at( 0 ), centralBodies.at( 0 ), bodyMap );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////             CREATE PROPAGATION SETTINGS            ////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Integration settings.
    const double fixedStepSize = 100000.0;
    std::shared_ptr< numerical_integrators::IntegratorSettings< > > integratorSettings =
            std::make_shared < numerical_integrators::IntegratorSettings < > >
            ( numerical_integrators::rungeKutta4, initialTime, fixedStepSize);

    // Termination condition
    std::function< Eigen::VectorXd( ) > spacecraftStateFunction =
            std::bind( &Body::getState, bodyMap.at( "Spacecraft" ) );
    std::shared_ptr< PropagationTerminationSettings > terminationSettings =
            std::make_shared< PropagationCustomTerminationSettings >(
                std::bind( &TerminationCondition, std::placeholders::_1, spacecraftStateFunction ) );

    // Define settings for propagation of translational dynamics.
    std::shared_ptr< TranslationalStatePropagatorSettings< double > > propagatorSettings =
            std::make_shared< TranslationalStatePropagatorSettings< double > >(
                centralBodies, accelerationModelMap, bodiesToPropagate, initialState, terminationSettings);

    // Define list of parameters to estimate.
    std::vector< std::shared_ptr<EstimatableParameterSettings > > parameterNames;
    parameterNames.push_back( std::make_shared< InitialTranslationalStateEstimatableParameterSettings< double > >(
                                  "Spacecraft", initialState, "Earth" ) );
    // Create parameters.
    std::shared_ptr< EstimatableParameterSet< double > > parametersToEstimate =
            createParametersToEstimate( parameterNames, bodyMap );

    // Print identifiers and indices of parameters to terminal.
    printEstimatableParameterEntries( parametersToEstimate );

    // Create simulation object and propagate dynamics.
    propagators::SingleArcVariationalEquationsSolver< > variationalEquationsSimulator(
                bodyMap, integratorSettings, propagatorSettings, parametersToEstimate, false,
                std::shared_ptr< numerical_integrators::IntegratorSettings< double > >( ), false, false );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////             PROPAGATE ORBIT            ////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::cout << "Propagating..." << std::endl;
    variationalEquationsSimulator.integrateVariationalAndDynamicalEquations(initialState, true);
    std::cout << "Propagation Finished!" << std::endl;

    std::map< double, Eigen::VectorXd> cr3bpPropagation =
            variationalEquationsSimulator.getDynamicsSimulator( )->getEquationsOfMotionNumericalSolution( );
    std::map< double, Eigen::MatrixXd > stateTransitionResult =
            variationalEquationsSimulator.getNumericalVariationalEquationsSolution( ).at( 0 );
    std::map< double, Eigen::MatrixXd > sensitivityResult =
            variationalEquationsSimulator.getNumericalVariationalEquationsSolution( ).at( 1 );

    // Transform to normalized corotating coordinates
    std::map< double, Eigen::Vector6d > cr3bpNormalisedCoRotatingFrame;
    for( std::map< double, Eigen::VectorXd >::iterator itr = cr3bpPropagation.begin( );
        itr != cr3bpPropagation.end( ); itr++ ){
        cr3bpNormalisedCoRotatingFrame[ itr->first ] = convertCartesianToCorotatingNormalizedCoordinates(
            gravitationalParameterPrimary, gravitationalParameterSecondary, distanceBodies, itr->second, itr->first);
        }

    std::cout << "Computing Lagrange point..." << std::endl;
    // Create LibrationPoint  object
    std::shared_ptr< root_finders::NewtonRaphsonCore< double > > rootFinder =
            std::make_shared< root_finders::NewtonRaphsonCore< double >>(1e-12, 100);
    LibrationPoint librationPoint = LibrationPoint(massParameter, rootFinder);
    // Select point
    librationPoint.computeLocationOfLibrationPoint(librationPoint.l1);
    // Retrieve point
    Eigen::Vector3d pointLagrange = librationPoint.getLocationOfLagrangeLibrationPoint();
    // Print result
    std::cout << "( " << pointLagrange[0] << ", " << pointLagrange[1] << ", " << pointLagrange[2] << " )" << std::endl;


    // Computing Newton-Rapson of function
    Eigen::VectorXd intialPeriodicCondition(7);
    intialPeriodicCondition << 0, 0.8, 0, 0, 0.1, 0;


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////        PROVIDE OUTPUT TO CONSOLE AND FILES           //////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    {
        std::cout << "Writing files..." << std::endl;
        // Write normalized corotating frame state
        input_output::writeDataMapToTextFile( cr3bpNormalisedCoRotatingFrame,
                                              "CR3BPnormalisedCoRotatingFrame.dat",
                                              tudat_applications::getOutputPath( ),
                                              "",
                                              std::numeric_limits< double >::digits10,
                                              std::numeric_limits< double >::digits10,
                                              "," );

        input_output::writeDataMapToTextFile( stateTransitionResult,
                                              "singlePerturbedSatelliteStateTransitionHistory.dat",
                                              tudat_applications::getOutputPath( ),
                                              "",
                                              std::numeric_limits< double >::digits10,
                                              std::numeric_limits< double >::digits10,
                                              "," );
    }

    // Final statement.
    // The exit code EXIT_SUCCESS indicates that the program was successfully executed.
    return EXIT_SUCCESS;
}
