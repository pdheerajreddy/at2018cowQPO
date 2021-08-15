from params import *
import niDataRedBasic as nid
import pandas as pd

if __name__ ==  '__main__':
	df = pd.read_csv(obsidlist, delim_whitespace=True, header=None, converters={i: str for i in range(0,1000)})
	obsids = df.iloc[:,0]

	coords = []
	for i in obsids:
		coords.append((ra, dec, str(i)))

	obj = nid.nicerpipe(workingdir, syspfilesdir, mpulist, pirange, trumpfilt, outdir_id)
	obj.run_parallel_pipe(coords)
