[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ySl5Eez0)

## Tunes Duel ♪: Music Recommendation System

This project explores probability and statistics through a music recommendation system. You will analyze user interactions with music recommendations using real song data and  ser ratings.

## Project Structure

```
├── README.md                    # This file
├── src/                         # Source code
│   ├── part1.py                 # Conditional Probability Modeling
│   ├── part2.py                 # User Variability Modeling  
│   ├── part3.py                 # Recommender Design
│   ├── part4.py                 # Monte Carlo Evaluation
│   └── recommender.py           # Main recommender for Tune Duel
└── data/                        # Dataset files (add your data here)
    ├── tracks.csv              # Song metadata (tracks)
    ├── ratings.csv             # User interactions
```

## Quick Start

1. **Add your data files** to the `data/` directory:
   - `tracks.csv` - Song metadata
   - `ratings.csv` - User interactions

2. **Run each part**:
   ```bash
   python src/part1.py
   python src/part2.py
   python src/part3.py
   python src/part4.py
   ```

3. **Test your recommender**:
   ```bash
   python src/recommender.py
   ```

## Project Parts

### Part 1: Conditional Probability Modeling (25 points)
- Compute conditional probabilities
- Use Bayes' rule
- Compare global vs. personal preferences

### Part 2: User Variability Modeling (25 points)
- Model time-to-5★ using geometric distribution
- Model user variability using Beta-geometric distribution
- Perform hypothesis testing between user groups

### Part 3: Recommender Design (20 points)
- Design two recommendation algorithms
- Prepare for Tune Duel competition

### Part 4: Monte Carlo Evaluation (20 points)
- Simulate user interactions
- Compute Hit@k, Average Rating, Time-to-5★
- Calculate confidence intervals
- Compare model performance

## Tune Duel Competition

Your `src/recommender.py` must implement the `query()` function:

```python
def query(song_ratings: List[Dict[str, Any]], topk: int = 5) -> List[Tuple[str, str]]:
    # Your implementation here
    pass
```

## Important Dates

- **Data Submission**: November 5, 2025
- **Code Submission**: December 14, 2025  
- **Report Submission**: December 21, 2025
- **Demo Sessions**: December 22-23, 2025

## Getting Started

1. **Create your personal session** by rating 10-20 songs from the track list
2. **Implement each part** following the TODO comments in the Python files
3. **Test your recommender** with the provided test function
4. **Submit your code** to GitHub and report to Moodle

## Tips
- Test your recommender thoroughly before submission
- Document your methods and results clearly

Good luck! 🎵
