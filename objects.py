import numpy as np

class Matrix(object):
  def __init__(self,shape:tuple[int,int],mat:np.ndarray) -> None:
    self.shape = shape
    self.mat = mat
    return
  @classmethod
  def from_default(cls,rows:int,cols:int,default:float):
    ndimarray = np.full((rows,cols),fill_value=default,dtype=float)
    mat = cls((rows,cols),ndimarray)
    return mat
  @classmethod
  def from_array(cls,rows:int,cols:int,result:np.ndarray):
    mat = cls((result.shape[0],result.shape[1]),result)
    return mat
  def __add__(self,other:np.ndarray):
    x = np.add(self.mat,other.mat)
    result = Matrix.from_array(self.shape[0],self.shape[1],x)
    return result
  def __mul__(self,other:np.ndarray):
    x = np.matmul(self.mat,other.mat)
    result = Matrix.from_array(self.shape[0],other.shape[1],x)
    return result
  def transpose(self) -> None:
    self.shape = (self.shape[1],self.shape[0])
    self.mat = np.transpose(self.mat)
    return
  def __repr__(self) -> str:
    return np.array2string(self.mat, formatter={'float_kind':lambda x: "%.2f" % x})
    
class Vector(Matrix):
  @classmethod
  def from_default(cls,rows:int,default:float):
    ndimarray = np.full(rows,fill_value=default,dtype=float)
    mat = cls((rows,1),ndimarray)
    return mat
  @classmethod
  def from_array(cls,result:np.ndarray):
    mat = cls((result.shape[0],1),result)
    return mat