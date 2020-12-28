# gmid2
* graphical models for influence diagrams 
* Junkyu Lee, Radu Marinescu, and Rina Dechter. "Submodel Decomposition Bounds for Influence Diagrams", AAAI 2021.
   
## Dependencies
```
$ conda create env --name gmid2 python=3.6
$ conda install numpy, networkx, sortedcontainers
```
* numpy version: 1.18
* networkx: 2.4

## Benchmark Problems in UAI file format
* benchmark data set under ```/gmid2/gmid2/benchmarks```
* file formats
  * .uai: same as uai file
  * .id: identity of variables and functions
  * .pvo: partial variable elimination ordering
  * .vo: variable elimination ordering

## Usage
* use scripts under ```/gmid2/gmid2/scripts```
```
$ PRJ_PATH=<path to gmid2>
$ python st_bte_bw.py $PRJ_PATH/benchmarks/synthetic/mpd1-4_2_2_5 2
$ python st_wmbmm_bw.py $PRJ_PATH/benchmarks/synthetic/mpd1-4_2_2_5 2
```
   
