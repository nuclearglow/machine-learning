# machine-learning

* `experiments` irgebwdwelche Skripte oder so
* `blechhuette`

# Install

```shell
conda install -c conda-forge ipdb
conda install -c conda-forge black
```

## project/ structure

data/
    * contains the data sources to be worked on
models/
    * contains models and other exported, loadable data (joblift pkl)
pipeline/
    * transformers/ - contains pipeline transfomers
    * pipeline.py - contains all pipelines, uses transformers

/ - the root project contains scripts
    01_get_data.py - download / procure data and save to data/
    02_data_visualization.py - scriupts to look at data (spyder)
    03_data_preparation.py - prepare data for modeling
    04_modeling.py - build models and optimize



# Tools

ls alias
https://github.com/ogham/exa

.bashrc

# Detect which `ls` flavor is in use
if ls --color > /dev/null 2>&1; then # GNU `ls`
	colorflag="--color"
	export LS_COLORS='no=00:fi=00:di=01;31:ln=01;36:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arj=01;31:*.taz=01;31:*.lzh=01;31:*.zip=01;31:*.z=01;31:*.Z=01;31:*.gz=01;31:*.bz2=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.jpg=01;35:*.jpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.avi=01;35:*.fli=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.ogg=01;35:*.mp3=01;35:*.wav=01;35:'
else # macOS `ls`
	colorflag="-G"
	export LSCOLORS='BxBxhxDxfxhxhxhxhxcxcx'
fi

# alias exa
alias ls='exa'
alias ll='exa --all --header --long --group --classify --git'

## Python

### IPDB Debugger

https://nornir.readthedocs.io/en/latest/howto/ipdb_how_to_inspect_complex_structures.html

### Run script in Python Shell

```shell
python
>>> exec(open('script.py').read())
```
