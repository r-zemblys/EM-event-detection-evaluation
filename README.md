# Evaluating Eye Movement Event Detection: A Review of the State of the Art
```bibtex
@Article{Startsev2021,
  author = {Mikhail Startsev and Raimondas Zemblys},
  title  = {Evaluating Eye Movement Event Detection: A Review of the State of the Art},
  year   = {2021},
}
```

## Run

```python
python run_calc.py
```

## List of publicly available annotated eye movement datasets
<p align='justify'>
List of publicly available annotated datasets to illustrate the variety of readily available material for algorithm development and evaluation. Duration reflects the amount of unique eye-tracking data (uniqueness judged based on file names); duration in parentheses -- the total amount of available annotated data (including undefined samples and taking into account several available annotations for a single recording). Sample distributions do not list proportion of undefined samples and samples annotated as noise, blinks and similar. Note that datasets might have different definitions of fixations, saccades, and other events.</p>

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Duration</th>
    <th>Set-up</th>
    <th>Sampling frequency</th>
    <th>Eye-tracker</th>
    <th>Sample distribution</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Lund2013<br>(Andersson et al., 2017)</td>
    <td>14.9 min<br>(18.6 min)</td>
    <td>Screen-based, pictures, moving dots and video clips</td>
    <td>500 Hz</td>
    <td>SMI Hi-Speed 1250</td>
    <td>46.49% Fixation<br>5.88% Saccade<br>3.34% PSO<br>41.60% Pursuit</td>
  </tr>
  <tr>
    <td colspan="6" align='justify'>Notes: two expert annotators, fully manually annotated, partial annotation overlap. Includes data that was used in other papers. Download from <a href="https://github.com/richardandersson/EyeMovementDetectorEvaluation" target="_blank" rel="noopener noreferrer">https://github.com/richardandersson/EyeMovementDetectorEvaluation</a></td>
  </tr>
  <tr>
    <td>IRF<br>(Zemblys et al., 2018)</td>
    <td>8.1 min</td>
    <td>Screen-based, fixate-saccade task</td>
    <td>1000 Hz</td>
    <td>EyeLink 1000Plus</td>
    <td>86.77% Fixation<br>5.65% Saccade<br>3.00% PSO</td>
  </tr>
  <tr>
    <td colspan="6" align='justify'>Notes: one expert annotator, fully manually annotated. Six participants, data from a replication study. Download from <a href="https://github.com/r-zemblys/irf" target="_blank" rel="noopener noreferrer">https://github.com/r-zemblys/irf</a></td>
  </tr>
  <tr>
    <td>MPIIEgoFixation<br>(Steil et al., 2018)</td>
    <td>24.2 min</td>
    <td>Head-mounted, unscripted daily life activities</td>
    <td>30 Hz</td>
    <td>Pupil Pro</td>
    <td>74.19% Fixation</td>
  </tr>
  <tr>
    <td colspan="6" align='justify'>Notes: frame-by-frame annotations of one annotator. Download from <a href="https://www.mpi-inf.mpg.de/MPIIEgoFixation" target="_blank" rel="noopener noreferrer">https://www.mpi-inf.mpg.de/MPIIEgoFixation</a></td>
  </tr>
  <tr>
    <td>humanFixationClassification<br>(Hooge et al., 2018)</td>
    <td>5.9 min<br>(70.4 min)</td>
    <td>Screen-based, pictures and search task</td>
    <td>300 Hz</td>
    <td>Tobii TX300</td>
    <td>71.82% Fixation</td>
  </tr>
  <tr>
    <td colspan="6" align='justify'>Notes: 12 expert annotators, fully manually annotated, all annotation data overlap. 10 adult free viewing and 60 infant search task trials. Download from <a href="https://github.com/dcnieho/humanFixationClassification" target="_blank" rel="noopener noreferrer">https://github.com/dcnieho/humanFixationClassification</a></td>
  </tr>
  <tr>
    <td>360EM<br>(Agtzidis et al., 2019)</td>
    <td>32.9 min</td>
    <td>Head-mounted, naturalistic 360&#176; videos</td>
    <td>120 Hz</td>
    <td>FOVE</td>
    <td>Primary labels: <br>75.15% Fixation<br>10.44% Saccade<br>9.76% Pursuit<br><br>Secondary labels:<br>0.81% OKN<br>27.64% VOR<br>15.84% OKN+VOR<br>1.47% Head pursuit</td>
  </tr>
  <tr>
    <td colspan="6" align='justify'>Notes: two stage annotations of one expert annotator after training and discussion session. First stage (primary labels and optokinetic nystagmus - OKN - or nystagmus) uses pre-labelled saccades and does not account for the head motion. Second stage (vestibulo-ocular reflex - VOR, VOR + OKN, Head pursuit) uses labels from the previous stage that are re-examined in the context of the eye-head coordination. Ca. 3.5 h of eye- and head-tracking recordings, ca. 16% annotated. Download from <a href="https://gin.g-node.org/ioannis.agtzidis/360_em_dataset" target="_blank" rel="noopener noreferrer">https://gin.g-node.org/ioannis.agtzidis/360_em_dataset</a></td>
  </tr>
  <tr>
    <td>GazeCom<br>(Startsev et al., 2019)</td>
    <td>4.7 h<br>(14.1 h)</td>
    <td>Screen-based, naturalistic video</td>
    <td>250 Hz</td>
    <td>EyeLink II</td>
    <td>73.96% Fixation<br>10.67% Saccade<br>9.83% Pursuit</td>
  </tr>
  <tr>
    <td colspan="6" align='justify'>Notes: manual annotations of one expert tie-breaking and adjusting labels of two novice annotators. Novice annotators (paid undergraduate students) used pre-labeled data and went through the data twice. Labels of novice annotators are available. Download from <a href="https://gin.g-node.org/ioannis.agtzidis/gazecom_annotations" target="_blank" rel="noopener noreferrer">https://gin.g-node.org/ioannis.agtzidis/gazecom_annotations</a></td>
  </tr>
  <tr>
    <td>Hollywood2EM<br>(Agtzidis et al., 2020)</td>
    <td>2.15 h<br>(4.3 h)</td>
    <td>Screen-based, movie clips</td>
    <td>500 Hz</td>
    <td>SMI Hi-Speed 1250</td>
    <td>59.46% Fixation<br>9.87% Saccade<br>26.54% Pursuit</td>
  </tr>
  <tr>
    <td colspan="6" align='justify'>Notes: manual annotations of pre-labeled data, two stage annotation (paid student followed by an expert coder). Labels of student annotator are available. Download from <a href="https://gin.g-node.org/ioannis.agtzidis/hollywood2_em" target="_blank" rel="noopener noreferrer">https://gin.g-node.org/ioannis.agtzidis/hollywood2_em</a></td>
  </tr>
  <tr>
    <td>Gaze-in-wild<br>(Kothatri et al., 2020)</td>
    <td>3.06 h<br>(4.15 h)</td>
    <td>Head mounted, naturalistic tasks</td>
    <td>300 Hz</td>
    <td>Pupil Labs + custom setup</td>
    <td>12.50% Fixation<br>7.12% Saccade<br>2.65% Pursuit<br>26.72% VOR</td>
  </tr>
  <tr>
    <td colspan="6" align='justify'>Notes: independent annotations of five trained annotators, ca. half of the data is annotated. Naturalistic tasks: indoor navigation, ball catching, object search, tea making. Download from <a href="http://www.cis.rit.edu/~rsk3900/gaze-in-wild" target="_blank" rel="noopener noreferrer">http://www.cis.rit.edu/~rsk3900/gaze-in-wild</a></td>
  </tr>
</tbody>
</table>


## Replicating paper results

1. Download Hollywood2EM (Agtzidis et al., 2020) dataset from [https://gin.g-node.org/ioannis.agtzidis/hollywood2_em](https://gin.g-node.org/ioannis.agtzidis/hollywood2_em)

2. Run the following to convert dataset to the required format:
```python
python misc/scripts/run_data_parse.py -root DATASET_ROOT -dataset hollywood2em --coder expert 
python misc/scripts/run_data_parse.py -root DATASET_ROOT -dataset hollywood2em --coder alg 
```
`DATASET_ROOT` is the directory where dataset was downloaded

3. Run evaluation script:
```python
python run_calc.py  -job assets/job_hollywood2.json
```
The result file will be save to `./results/job_hollywood2.csv`. This can take around 3 hours or more, depending on the computer

4. Run data analysis script:
```python
python misc/scripts/analyse_results.py
```
Resulting plots will be saved to `./results/job_hollywood2/result-plots`

To get matcher example plots run:
```python
python run_calc.py -job assets/job_plot.json
```
Resulting plots will be saved to `./results/match-plots`

## ETRA2022 presentation slides

https://docs.google.com/presentation/d/1mBjzA4piXZzgS-6g4V_yBznfnwZ1wLI29cj2j7mShVE/edit?usp=sharing
