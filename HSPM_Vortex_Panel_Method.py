# Major Libraries

import numpy
import matplotlib
import matplotlib.pyplot

# Input Data

freestreamVelocity = 10
angleOfAttack = numpy.radians(
    10
)

# Airfoil Data

horizontalCoordinates = numpy.flip(
    numpy.loadtxt(
        'NACA_4412.dat'
    )[:,0],
    axis = 0
)

verticalCoordinates = numpy.flip(
    numpy.loadtxt(
        'NACA_4412.dat'
    )[:,1],
    axis = 0
)

# Normalize Horizontal and Vertical Coordinates

chordLength = horizontalCoordinates[0]
horizontalCoordinates = horizontalCoordinates/chordLength
verticalCoordinates = verticalCoordinates/chordLength

# Quantity Numbers

vorticesQuantity = len(
    horizontalCoordinates
)

panelsQuantity = vorticesQuantity-1

# Panels Midpoints Coordinates

horizontalPanelsMidpointsCoordinates = numpy.zeros(
    (
        panelsQuantity
    ),
    dtype = float
)

verticalPanelsMidpointsCoordinates = numpy.zeros(
    (
        panelsQuantity
    ),
    dtype = float
)

for mainCounter in range(panelsQuantity):
    horizontalPanelsMidpointsCoordinates[mainCounter] = (
        horizontalCoordinates[mainCounter]+horizontalCoordinates[mainCounter+1]
    )/2
    verticalPanelsMidpointsCoordinates[mainCounter] = (
        verticalCoordinates[mainCounter]+verticalCoordinates[mainCounter+1]
    )/2

# Panels Angles

panelsAngles = numpy.zeros(
    (
        panelsQuantity
    ),
    dtype = float
)

for mainCounter in range(panelsQuantity):
    panelsAngles[mainCounter] = numpy.arctan2(
        verticalCoordinates[mainCounter+1]-verticalCoordinates[mainCounter],
        horizontalCoordinates[mainCounter+1]-horizontalCoordinates[mainCounter]
    )

# Position Matrix

positionMatrix = numpy.zeros(
    (
        panelsQuantity,
        vorticesQuantity
    ),
    dtype = float
)

for mainCounter in range(panelsQuantity):
    for secondaryCounter in range(vorticesQuantity):
        positionMatrix[mainCounter,secondaryCounter] = numpy.sqrt(
            numpy.power(
                horizontalPanelsMidpointsCoordinates[mainCounter]-horizontalCoordinates[secondaryCounter],
                2
            )+numpy.power(
                verticalPanelsMidpointsCoordinates[mainCounter]-verticalCoordinates[secondaryCounter],
                2
            )
        )

# Relative Panels Angles

relativePanelsAngles = numpy.zeros(
    (
        panelsQuantity,
        panelsQuantity
    ),
    dtype = float
)

for mainCounter in range(panelsQuantity):
    for secondaryCounter in range(panelsQuantity):
            relativePanelsAngles[mainCounter,secondaryCounter] = (
                 numpy.mod(
                    numpy.arctan2(
                        verticalPanelsMidpointsCoordinates[mainCounter]-verticalCoordinates[secondaryCounter+1],
                        horizontalPanelsMidpointsCoordinates[mainCounter]-horizontalCoordinates[secondaryCounter+1]
                    )-panelsAngles[secondaryCounter]+numpy.pi,
                    2*numpy.pi
                )-numpy.pi
            )-(
                 numpy.mod(
                    numpy.arctan2(
                        verticalPanelsMidpointsCoordinates[mainCounter]-verticalCoordinates[secondaryCounter],
                        horizontalPanelsMidpointsCoordinates[mainCounter]-horizontalCoordinates[secondaryCounter]
                    )-panelsAngles[secondaryCounter]+numpy.pi,
                    2*numpy.pi
                )-numpy.pi
            )

# Normal and Tangential Coefficient Matrices

normalCoefficientMatrix = numpy.zeros(
    (
        panelsQuantity,
        panelsQuantity
    ),
    dtype = float
)

tangentialCoefficientMatrix = numpy.zeros(
    (
        panelsQuantity,
        panelsQuantity
    ),
    dtype = float
)

for mainCounter in range(panelsQuantity):
    for secondaryCounter in range(panelsQuantity):
        if mainCounter == secondaryCounter:
            normalCoefficientMatrix[mainCounter,secondaryCounter] = 0.5
            tangentialCoefficientMatrix[mainCounter,secondaryCounter] = 0
        if mainCounter != secondaryCounter:
            normalCoefficientMatrix[mainCounter,secondaryCounter] = (
                numpy.sin(
                    panelsAngles[mainCounter]-panelsAngles[secondaryCounter]
                )*numpy.log(
                    positionMatrix[mainCounter,secondaryCounter+1]/positionMatrix[mainCounter,secondaryCounter]
                )+numpy.cos(
                    panelsAngles[mainCounter]-panelsAngles[secondaryCounter]
                )*relativePanelsAngles[mainCounter,secondaryCounter]
            )/2/numpy.pi
            tangentialCoefficientMatrix[mainCounter,secondaryCounter] = (
                numpy.sin(
                    panelsAngles[mainCounter]-panelsAngles[secondaryCounter]
                )*relativePanelsAngles[mainCounter,secondaryCounter]-numpy.cos(
                    panelsAngles[mainCounter]-panelsAngles[secondaryCounter]
                )*numpy.log(
                    positionMatrix[mainCounter,secondaryCounter+1]/positionMatrix[mainCounter,secondaryCounter]
                )
            )/2/numpy.pi

# Normal and Tangential Augmenting Matrices

normalAugmentingMatrix = numpy.zeros(
    (
        panelsQuantity,
        panelsQuantity
    ),
    dtype = float
)

tangentialAugmentingMatrix = numpy.zeros(
    (
        panelsQuantity,
        panelsQuantity
    ),
    dtype = float
)

normalAugmentingMatrix = -tangentialCoefficientMatrix
tangentialAugmentingMatrix = normalCoefficientMatrix

# Coefficient Matrix

coefficientMatrix = numpy.zeros(
    (
        vorticesQuantity,
        vorticesQuantity
    ),
    dtype = float
)

for mainCounter in range(panelsQuantity):
    for secondaryCounter in range(panelsQuantity):
        coefficientMatrix[mainCounter,secondaryCounter] = normalCoefficientMatrix[mainCounter,secondaryCounter]
        coefficientMatrix[panelsQuantity,mainCounter] = tangentialCoefficientMatrix[0,mainCounter]+tangentialCoefficientMatrix[panelsQuantity-1,mainCounter]
    coefficientMatrix[mainCounter,-1] = numpy.sum(
        normalAugmentingMatrix[mainCounter,:]
    )

coefficientMatrix[panelsQuantity,panelsQuantity] = numpy.sum(
    tangentialAugmentingMatrix[0,:]+tangentialAugmentingMatrix[panelsQuantity-1,:]
)

# Augmenting Vector

augmentingVector = numpy.zeros(
    (
        vorticesQuantity
    ),
    dtype = float
)

for mainCounter in range(vorticesQuantity):
    if mainCounter != panelsQuantity:
        augmentingVector[mainCounter] = -freestreamVelocity*numpy.sin(
            angleOfAttack-panelsAngles[mainCounter]
        )
    if mainCounter == panelsQuantity:
        augmentingVector[panelsQuantity] = -freestreamVelocity*numpy.cos(
            angleOfAttack-panelsAngles[0]
        )-freestreamVelocity*numpy.cos(
            angleOfAttack-panelsAngles[panelsQuantity-1]
        )

# Source Strengths and Vortex Strength

sourceStrengths = numpy.linalg.solve(
    coefficientMatrix,
    augmentingVector
)[0:panelsQuantity]

vortexStrength = numpy.linalg.solve(
    coefficientMatrix,
    augmentingVector
)[panelsQuantity]

# Tangential Velocities

tangentialVelocities = numpy.zeros(
    (
        panelsQuantity
    ),
    dtype = float
)

for mainCounter in range(panelsQuantity):
    tangentialVelocities[mainCounter] = numpy.dot(
        tangentialCoefficientMatrix[mainCounter,:],
        sourceStrengths[:]
    )+vortexStrength*numpy.sum(
        tangentialAugmentingMatrix[mainCounter,:]
    )+freestreamVelocity*numpy.cos(
        angleOfAttack-panelsAngles[mainCounter]
    )

# Pressure Coefficient

pressureCoefficient = numpy.zeros(
    (
        panelsQuantity
    ),
    dtype = float
)

for mainCounter in range(panelsQuantity):
    pressureCoefficient[mainCounter] = 1-numpy.power(
        tangentialVelocities[mainCounter]/freestreamVelocity,
        2
    )

# Pressure Coefficient Distribution over Lower and Upper Surfaces

# Even Number of Vortices

if vorticesQuantity % 2 == 0:

    halfPosition = int(
        vorticesQuantity/2
    )

    lowerSurfacePressureCoefficient = pressureCoefficient[:halfPosition]
    upperSurfacePressureCoefficient = pressureCoefficient[halfPosition:]

    lowerSurfaceHorizontalMidpointCoordinates = horizontalPanelsMidpointsCoordinates[:halfPosition]
    upperSurfaceHorizontalMidpointCoordinates = horizontalPanelsMidpointsCoordinates[halfPosition:]

# Odd Number of Vortices

if vorticesQuantity % 2 != 0:

    halfPosition = int(
        (
            vorticesQuantity-1
        )/2
    )
    
    lowerSurfacePressureCoefficient = pressureCoefficient[:halfPosition]
    upperSurfacePressureCoefficient = pressureCoefficient[halfPosition:]

    lowerSurfaceHorizontalMidpointCoordinates = horizontalPanelsMidpointsCoordinates[:halfPosition]
    upperSurfaceHorizontalMidpointCoordinates = horizontalPanelsMidpointsCoordinates[halfPosition:]

# Rotated Coordinates

rotatedHorizontalCoordinates = numpy.zeros(
    (
        vorticesQuantity
    ),
    dtype = float
)

rotatedVerticalCoordinates = numpy.zeros(
    (
        vorticesQuantity
    ),
    dtype = float
)

for mainCounter in range(vorticesQuantity):
    rotatedHorizontalCoordinates[mainCounter] = horizontalCoordinates[mainCounter]*numpy.cos(
        -angleOfAttack
    )-verticalCoordinates[mainCounter]*numpy.sin(
        -angleOfAttack
    )
    rotatedVerticalCoordinates[mainCounter] = horizontalCoordinates[mainCounter]*numpy.sin(
        -angleOfAttack
    )+verticalCoordinates[mainCounter]*numpy.cos(
        -angleOfAttack
    )

# Lift Coefficient

liftCoefficient = numpy.trapz(
    numpy.flip(
        lowerSurfacePressureCoefficient
    ),
    numpy.flip(
        lowerSurfaceHorizontalMidpointCoordinates
    )
)-numpy.trapz(
    upperSurfacePressureCoefficient,
    upperSurfaceHorizontalMidpointCoordinates
)

# Graphics

# General

figure, axes = matplotlib.pyplot.subplots(
    2,
    1,
    figsize = (
        6,
        8
    )
)

figure.patch.set_facecolor(
    'black'
)

matplotlib.pyplot.subplots_adjust(
    left = 0.2,
    right = 0.9,
    top = 0.9,
    bottom = 0.1
)

# Pressure Coefficient

firstAxis = axes[0]

firstAxis.set_facecolor(
    'black'
)

firstAxis.plot(
    numpy.flip(
        lowerSurfaceHorizontalMidpointCoordinates
    ),
    numpy.flip(
        lowerSurfacePressureCoefficient
    ),
    color = 'cyan',
    linewidth = 1,
    label = 'Pressure Distribution along Lower Surface'
)

firstAxis.plot(
    upperSurfaceHorizontalMidpointCoordinates,
    upperSurfacePressureCoefficient,
    color = 'magenta',
    linewidth = 1,
    label = 'Pressure Distribution along Upper Surface'
)

firstAxis.invert_yaxis()

firstAxis.set_xlabel(
    'Chordwise Relative Position, %',
    color = 'white',
    fontsize = 10
)

firstAxis.set_ylabel(
    'Pressure Coefficient',
    color = 'white',
    fontsize = 10
)

firstAxis.tick_params(
    colors = 'white',
    labelsize = 10
)

firstAxis.legend(
    facecolor = 'black',
    edgecolor = 'black',
    labelcolor = 'white',
    fontsize = 10
)

firstAxis.set_title(
    f'Angle of Attack: {numpy.degrees(angleOfAttack)}\u00B0, Lift Coefficient: {liftCoefficient:.4f}',
    color = 'white',
    fontsize = 10
)

firstAxis.spines['bottom'].set_color(
    'white'
)

firstAxis.spines['top'].set_color(
    'black'
)

firstAxis.spines['left'].set_color(
    'white'
)

firstAxis.spines['right'].set_color(
    'black'
)

firstAxis.set_xlim(
    0,
    1
)

# Airfoil

secondAxis = axes[1]

secondAxis.set_facecolor(
    'black'
)

secondAxis.plot(
    rotatedHorizontalCoordinates,
    rotatedVerticalCoordinates,
    color = 'yellow',
    linewidth = 1
)

secondAxis.set_aspect(
    'equal',
    adjustable = 'box'
)

secondAxis.set_xlabel(
    'Horizontal Position, %',
    color = 'white',
    fontsize = 10
)

secondAxis.set_ylabel(
    'Vertical Position, %',
    color = 'white',
    fontsize = 10
)

secondAxis.tick_params(
    colors = 'white',
    labelsize = 10
)

secondAxis.spines['bottom'].set_color(
    'white'
)

secondAxis.spines['top'].set_color(
    'black'
)

secondAxis.spines['left'].set_color(
    'white'
)

secondAxis.spines['right'].set_color(
    'black'
)

secondAxis.set_xlim(
    0,
    rotatedHorizontalCoordinates[panelsQuantity]
)

matplotlib.pyplot.show()