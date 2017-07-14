local Replicate, parent = torch.class('nn.Replicate','nn.Module')

function Replicate:__init(nf, dim, ndim)
   parent.__init(self)
   self.nfeatures = nf
   self.dim = dim or 1
   self.ndim = ndim
   assert(self.dim > 0, "Can only replicate across positive integer dimensions.")
end

function Replicate:updateOutput(input)
   local start = sys.clock()
   self.dim = self.dim or 1 --backwards compatible
   assert(
      self.dim <= input:dim()+1,
      "Not enough input dimensions to replicate along dimension " ..
      tostring(self.dim) .. ".")
   local batchOffset = self.ndim and input:dim() > self.ndim and 1 or 0
   local rdim = self.dim + batchOffset
   local sz = torch.LongStorage(input:dim()+1)
   sz[rdim] = self.nfeatures
   for i = 1,input:dim() do
      local offset = 0
      if i >= rdim then
         offset = 1
      end
      sz[i+offset] = input:size(i)
   end
   local st = torch.LongStorage(input:dim()+1)
   st[rdim] = 0
   for i = 1,input:dim() do
      local offset = 0
      if i >= rdim then
         offset = 1
      end
      st[i+offset] = input:stride(i)
   end
   self.output:set(input:storage(),input:storageOffset(),sz,st)
   self.t1 = self.t1 + sys.clock() - start
   self.count = self.count + 1
   if self.count == 100 then
      print("Replicate_F = ", self.t1)
      print("Replicate_B = ", self.t2)
      self.t1 = 0
      self.t2 = 0
      self.count = 0
   end
   return self.output
end

function Replicate:updateGradInput(input, gradOutput)
   local start = sys.clock()
   self.gradInput:resizeAs(input):zero()
   local batchOffset = self.ndim and input:dim() > self.ndim and 1 or 0
   local rdim = self.dim + batchOffset
   local sz = torch.LongStorage(input:dim()+1)
   sz[rdim] = 1
   for i = 1,input:dim() do
      local offset = 0
      if i >= rdim then
         offset = 1
      end
      sz[i+offset] = input:size(i)
   end
   local gradInput = self.gradInput:view(sz)
   gradInput:sum(gradOutput, rdim)
   self.t2 = self.t2 + sys.clock() - start
   return self.gradInput
end
