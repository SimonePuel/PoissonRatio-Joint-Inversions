"""
This code defines a class to read and write a DOLFIN Function
to a .xdmf file.

@ author: Simone Puel (spuel@utexas.edu)
"""


# Import libraries
import dolfin as dl

# Create a class to read and write a DOLFIN function to .xdmf file
class FunctionsIO:
    # Write to .xdmf file
    @classmethod
    def write(cls, comm, filename, function_names, function_list):
        with dl.XDMFFile(comm, filename + ".xdmf") as fid:
            for f, fn in zip(function_list, function_names):
                    fid.write_checkpoint(f, fn, append=True)
              
    # Read a .xdmf file        
    @classmethod
    def read(cls, comm, filename, Vh, function_names):
        functions = []
        with dl.XDMFFile(comm, filename + ".xdmf") as fid:
            for fn in function_names:
                f = dl.Function(Vh)
                fid.read_checkpoint(f, fn)
                functions.append(f)
        
        return functions