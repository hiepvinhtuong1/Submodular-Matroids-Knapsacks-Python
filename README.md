# Submodular-Matroids-Knapsacks-Python1

This repository contains Python implementations of submodular optimization algorithms with matroid and knapsack constraints, including Greedy, Repeated Greedy, Simultaneous Greedy, FANTOM, SPROUT, and SPROUT++. The project includes scripts to run experiments on movie recommendation datasets and graph cut problems, as well as visualization scripts to plot results.

## Project Structure

-   `data/`: Contains datasets (e.g., `movie_info.csv` for movie recommendation experiments).
-   `experiments/`: Contains experiment scripts.
    -   `movie_recommendation/`: Scripts for movie recommendation experiments (`two_knapsacks.py`, `three_knapsacks.py`, `mu.py`, `tc.py`).
    -   `parametric-sensitivity-analysis/`: Scripts for sensitivity analysis (`tc.py` for graph cut experiments).
-   `submodular_greedy/`: Contains the core implementation of optimization algorithms.
    -   `algorithms/`: Implementation of algorithms (`greedy.py`, `repeated_greedy.py`, `simultaneous_greedys.py`, `fantom.py`, `sprout.py`).
    -   `utils/`: Helper functions (`helper_funs.py`).
-   `visualization/`: Contains scripts for plotting results (`draw.py`).
-   `requirements.txt`: Lists the required Python libraries.

## Requirements

-   **Python**: Version 3.8 or 3.9 is recommended.
-   **Operating System**: Windows, macOS, or Linux.
-   **Dependencies**: Listed in `requirements.txt` (see installation instructions below).

## Installation Instructions

### Step 1: Install Python

Ensure Python 3.8 or 3.9 is installed on your system.

1. **Check if Python is installed**:

    ```bash
    python --version
    ```

    or

    ```bash
    python3 --version
    ```

    You should see a version like `Python 3.9.5`. If not, proceed to install Python.

2. **Install Python** (if not already installed):

    - Download Python from [python.org](https://www.python.org/downloads/).
    - Install Python:
        - On Windows: Run the installer and ensure you check **"Add Python to PATH"** before clicking "Install Now".
        - On macOS/Linux: Follow the installation instructions (e.g., use `brew` on macOS or `apt` on Ubuntu).
    - Verify the installation:
        ```bash
        python --version
        ```

3. **Install `pip`** (Python package manager):
    - `pip` is usually installed with Python. Check with:
        ```bash
        pip --version
        ```
    - If `pip` is not installed, download `get-pip.py` from [bootstrap.pypa.io/get-pip.py](https://bootstrap.pypa.io/get-pip.py) and run:
        ```bash
        python get-pip.py
        ```

### Step 2: Set Up a Virtual Environment

Using a virtual environment helps isolate project dependencies and avoid conflicts.

1. **Navigate to the project directory**:

    ```bash
    cd /path/to/Submodular-Matroids-Knapsacks-Python1
    ```

    For example:

    ```bash
    cd D:\Nghien cuu\Submodular-Matroids-Knapsacks-Python1
    ```

2. **Create a virtual environment**:

    ```bash
    python -m venv venv
    ```

    This creates a folder named `venv` in your project directory.

3. **Activate the virtual environment**:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
        After activation, you should see `(venv)` in your command prompt, e.g., `(venv) D:\Nghien cuu\Submodular-Matroids-Knapsacks-Python1>`.

### Step 3: Install Required Libraries

The project depends on several Python libraries. You can install them using `pip`.

1. **Install dependencies from `requirements.txt`** (if available):
   If a `requirements.txt` file exists in the project directory, run:

    ```bash
    pip install -r requirements.txt
    ```

2. **Install libraries manually** (if `requirements.txt` is not available):
   The project requires the following libraries:

    - `matplotlib`: For plotting (`draw.py`).
    - `numpy`: For numerical computations.
    - `networkx`: For graph operations (`tc.py`).
    - `scipy`: For distance calculations (`two_knapsacks.py`, etc.).
    - `pandas`: For data processing (`two_knapsacks.py`, etc.).

    Install them with:

    ```bash
    pip install matplotlib numpy networkx scipy pandas
    ```

3. **Verify installed libraries**:
   Check the installed libraries:
    ```bash
    pip list
    ```
    You should see `matplotlib`, `numpy`, `networkx`, `scipy`, and `pandas` in the list.

## Running the Code

### 1. Run Visualization Script (`draw.py`)

This script generates a plot comparing the performance of different algorithms.

1. **Navigate to the `visualization` directory**:

    ```bash
    cd visualization
    ```

2. **Run the script**:
    ```bash
    python draw.py
    ```
    - This will generate a file named `plot.png` in the `visualization` directory.
    - The plot shows the objective value vs. knapsack budget for five algorithms: DSSGS, RP_Greedy, Greedy, FANTOM, and SPROUT++.

### 2. Run Movie Recommendation Experiments

These scripts run optimization algorithms on a movie dataset (`movie_info.csv`).

1. **Navigate to the `experiments/movie_recommendation` directory**:

    ```bash
    cd experiments/movie_recommendation
    ```

2. **Run `two_knapsacks.py` (2 knapsack constraints)**:

    ```bash
    python two_knapsacks.py
    ```

    - This script runs five algorithms (Greedy, Repeated Greedy, Simultaneous Greedy, FANTOM, SPROUT++) with 2 knapsack constraints (rating and year).

3. **Run `three_knapsacks.py` (3 knapsack constraints)**:

    ```bash
    python three_knapsacks.py
    ```

    - Similar to `two_knapsacks.py`, but with 3 knapsack constraints (rating, year, runtime).

4. **Run `mu.py` (sensitivity analysis for `mu` parameter)**:
    ```bash
    python mu.py
    ```
    - This script runs SPROUT++ with varying values of the `mu` parameter.

### 3. Run Graph Cut Experiments (`tc.py`)

This script runs optimization algorithms on a graph cut problem using an Erdos-Renyi graph.

1. **Navigate to the `experiments/parametric-sensitivity-analysis` directory**:

    ```bash
    cd experiments/parametric-sensitivity-analysis
    ```

2. **Run `tc.py`**:
    ```bash
    python tc.py
    ```
    - This script runs five algorithms (Greedy, Repeated Greedy, Simultaneous Greedy, FANTOM, SPROUT++) on a graph cut problem with no constraints.

## Notes

-   **Font Issue (`Times New Roman`)**:

    -   The `draw.py` script uses the `'Times New Roman'` font for labels and legends. If this font is not available on your system, you may see a warning or the font may not render correctly. To resolve this:
        -   Ensure `'Times New Roman'` is installed on your system.
        -   Alternatively, remove the `fontfamily` and `prop={'family': 'Times New Roman'}` parameters to use the default font:
            ```python
            plt.xlabel('Knapsack budget', fontsize=20)
            plt.ylabel('Objective value', fontsize=26)
            plt.legend(loc='lower right', fontsize=14)
            ```

-   **Dependencies**:

    -   If you encounter errors about missing libraries, install the missing library using:
        ```bash
        pip install <library-name>
        ```
    -   To generate a `requirements.txt` file for your current environment:
        ```bash
        pip freeze > requirements.txt
        ```

-   **Deactivate Virtual Environment**:
    -   When you're done, deactivate the virtual environment:
        ```bash
        deactivate
        ```

## Troubleshooting

-   **Python Version Issues**:

    -   Ensure you're using Python 3.8 or 3.9. Check with:
        ```bash
        python --version
        ```
    -   If needed, install a compatible version from [python.org](https://www.python.org/downloads/).

-   **Library Version Conflicts**:

    -   If you encounter errors due to library version conflicts, try installing specific versions:
        ```bash
        pip install matplotlib==3.5.1 numpy==1.21.5 networkx==2.6.3 scipy==1.7.3 pandas==1.3.5
        ```

-   **File Not Found**:
    -   Ensure the `movie_info.csv` file exists in the `data/` directory for movie recommendation experiments.
    -   Ensure all paths in the scripts are correct for your system.

## Contact

If you encounter any issues or need further assistance, please open an issue in this repository or contact the maintainers.
