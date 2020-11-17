# Create a test set

RUN python animal/arenas/create_test_set.py --target-dir /path/to/target/directory

to create a basic test set that a successfully trained agent is expected to complete.
In the test set most of the features are fixed.
It includes:

    - 10 c1 arenas: solve environment with green, orange and red balls of equal size.
    - 10 c1 arenas (special): only one small (0.5) red ball
    - 10 c2 arenas: solve environment with green, orange and red balls of different size.
    - 10 c3 arenas: solve environment with multiple objects.
    - 10 c3 arenas (special): only walls like in the cl animalai example.
    - 10 c4 arenas: solve environment with red and orange zones.
    - 10 c5 arenas: solve environment were climbing up ramps is required to get goals.
    - 10 c6 arenas: solve environment with multiple objects and random colors.
    - 10 c6 arenas (special): only randomly colored walls like in the cl animalai example.
    - 10 c7 arenas: solve environment with multiple objects and blackouts.
    - 10 c8 arenas: mazes.
    - 10 c9 arenas: choice situation.

# Create a train set

RUN python animal/arenas/create_train_set.py --target-dir /path/to/target/directory

to create a train with many different types of arenas. In the train set most of the features are random.
Note that c8 and c9 are probably different from real c8 and c9 arenas.

It includes:

    - 500 c1 arenas: solve environment with green, orange and red balls of equal size.
    - 200 c1 arenas (special): only one small (0.5) red ball
    - 500 c2 arenas: solve environment with green, orange and red balls of different size.
    - 500 c3 arenas: solve environment with multiple objects.
    - 500 c3 arenas (special): only walls like in the cl animalai example.
    - 500 c4 arenas: solve environment with red and orange zones.
    - 500 c5 arenas: solve environment were climbing up ramps is required to get goals.
    - 500 c6 arenas: solve environment with multiple objects and random colors.
    - 500 c6 arenas (special): only randomly colored walls like in the cl animalai example.
    - 500 c7 arenas: solve environment with multiple objects and blackouts.
    - 200 c8 arenas: mazes.
    - 500 c9 arenas: choice situation.

