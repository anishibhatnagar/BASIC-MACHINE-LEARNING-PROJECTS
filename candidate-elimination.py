#Candidate
import pandas as pd
import numpy as np
from google.colab import files
data=pd.read_csv(next(iter(files.upload())))
print("Dataset:\n",data)
def ce(concept,target):
  sh=concept[0].copy()
  print("\nsh:",sh)
  gh=[['?' for _ in range(len(sh))] for _ in range(len(sh))]
  print("gh:",gh)
  for i,h in enumerate(concept):
    if target[i]=='yes':
      for x in range(len(sh)):
        if h[x]!=sh[x]:
          sh[x]='?'
          gh[x][x]='?'
      print("\nsteps in cea(",i+1,"):")
      print("sh:",sh)
      print("gh:",gh)
    if target[i]=='no':
      for x in range(len(sh)):
        if h[x]!=sh[x]:
          gh[x][x]=sh[x]
        else:
          gh[x][x]='?'
      print("\nsteps in cea(",i+1,")")
      print("sh:",sh)
      print("gh:",gh)
  indices=[i for i,val in enumerate(gh) if val==['?']*len(sh)]
  for i in indices:
    gh.remove(['?']*len(sh))
  return sh,gh
sf,gf=ce(np.array(data)[:,:-1],np.array(data)[:,-1])
print("\nfinal specific hyp:",sf)
print("final genaral hyp:",gf)
