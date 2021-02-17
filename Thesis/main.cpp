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

#include <Tudat/SimulationSetup/tudatSimulationHeader.h>
#include <Tudat/InputOutput/basicInputOutput.h>
#include "Thesis/applicationOutput.h"
#include "Tudat/Astrodynamics/Ephemerides/approximatePlanetPositions.h"
#include "Tudat/Astrodynamics/Gravitation/unitConversionsCircularRestrictedThreeBodyProblem.h"
#include "Tudat/Astrodynamics/OrbitDetermination/EstimatableParameters/estimatableParameter.h"

int main( )
{
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////            USING STATEMENTS              //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    using namespace tudat;
    using namespace tudat::input_output;
    using namespace tudat::simulation_setup;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////     CREATE ENVIRONMENT AND VEHICLE       //////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Load Spice kernels.
    spice_interface::loadStandardSpiceKernels( );

    // Define primaries
    string primary = "Sun";
    string secondary = "Jupiter";

    // // Set simulation time settings.
    double initialTime = 0.0;

    // Global characteristics of the problem
    double distanceBodies = 778.0e9;

    // Initialise the spacecraft state (B. Taylor, D. (1981). Horseshoe periodic orbits in the restricted problem of three bodies
    // for a sun-Jupiter mass ratio. Astronomy and Astrophysics. 103. 288-294.)
    Eigen::Vector6d initialState = Eigen::Vector6d::Zero();
    initialState[0] = - 7.992e11;
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

    NamedBodyMap bodyMap = propagators::setupBodyMapCR3BP(
                distanceBodies, primary, secondary, "Spacecraft" );

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
    double finalTime = tudat::circular_restricted_three_body_problem::convertDimensionlessTimeToDimensionalTime(
                29.2386 * ( 2.0 * mathematical_constants::PI ), gravitationalParameterPrimary, gravitationalParameterSecondary, distanceBodies);

    // Compute mass parameter
    double massParameter = circular_restricted_three_body_problem::computeMassParameter(
                gravitationalParameterPrimary, gravitationalParameterSecondary );
    std::cout << "Mass Parameter: " << massParameter << std::endl;

    /// CR3BP problem
    // Create acceleration map.
    basic_astrodynamics::AccelerationMap accelerationModelMap = propagators::setupAccelerationMapCR3BP(
            primary, secondary, bodiesToPropagate.at( 0 ), centralBodies.at( 0 ), bodyMap );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////             CREATE PROPAGATION SETTINGS            ////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    const double fixedStepSize = 100000.0;
    std::shared_ptr< numerical_integrators::IntegratorSettings< > > integratorSettings =
            std::make_shared < numerical_integrators::IntegratorSettings < > >
            ( numerical_integrators::rungeKutta4, initialTime, fixedStepSize);

    std::shared_ptr< propagators::TranslationalStatePropagatorSettings< double > > propagatorSettings =
            std::make_shared< propagators::TranslationalStatePropagatorSettings< double > >(
                centralBodies, accelerationModelMap, bodiesToPropagate, initialState, finalTime );

    // Define list of parameters to estimate.
    std::vector< std::shared_ptr<estimatable_parameters::EstimatableParameterSettings > > parameterNames;
    parameterNames.push_back( std::make_shared< estimatable_parameters::InitialTranslationalStateEstimatableParameterSettings< double > >(
                                  "Spacecraft", initialState, "Earth" ) );
    // Create parameters
    std::shared_ptr< estimatable_parameters::EstimatableParameterSet< double > > parametersToEstimate =
            createParametersToEstimate( parameterNames, bodyMap );

    // Print identifiers and indices of parameters to terminal.
    printEstimatableParameterEntries( parametersToEstimate );

    // Create simulation object and propagate dynamics.
    propagators::SingleArcVariationalEquationsSolver< > variationalEquationsSimulator(
                bodyMap, integratorSettings, propagatorSettings, parametersToEstimate, true,
                std::shared_ptr< numerical_integrators::IntegratorSettings< double > >( ), false, true );

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////             PROPAGATE ORBIT            ////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::map< double, Eigen::VectorXd> cr3bpPropagation =
            variationalEquationsSimulator.getDynamicsSimulator( )->getEquationsOfMotionNumericalSolution( );
    std::map< double, Eigen::MatrixXd > stateTransitionResult =
            variationalEquationsSimulator.getNumericalVariationalEquationsSolution( ).at( 0 );
    /*std::map< double, Eigen::MatrixXd > sensitivityResult =
            variationalEquationsSimulator.getNumericalVariationalEquationsSolution( ).at( 1 );*/

    // Transform to normalized corotating coordinates
    std::map< double, Eigen::Vector6d > cr3bpNormalisedCoRotatingFrame;
    for( std::map< double, Eigen::VectorXd >::iterator itr = cr3bpPropagation.begin( );
        itr != cr3bpPropagation.end( ); itr++ ){
        cr3bpNormalisedCoRotatingFrame[ itr->first ] = tudat::circular_restricted_three_body_problem::convertCartesianToCorotatingNormalizedCoordinates(
            gravitationalParameterPrimary, gravitationalParameterSecondary, distanceBodies, itr->second, itr->first);
        }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////        PROVIDE OUTPUT TO CONSOLE AND FILES           //////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    {
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
