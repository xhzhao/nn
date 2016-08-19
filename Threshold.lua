local Threshold, parent = torch.class('nn.Threshold','nn.Module')

function Threshold:__init(th,v,ip)
   parent.__init(self)
   self.threshold = th or 1e-6
   self.val = v or 0
   if (th and type(th) ~= 'number') or (v and type(v) ~= 'number') then
      error('nn.Threshold(threshold, value)')
   end
   -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
   self:validateParameters()
   if sys then
      self.timerEnable = sys.timerEnable or false
   else
      self.timerEnable = false
   end
   self.timeForward = 0
   self.timeBackward = 0
   self.cnt = 0

end

function Threshold:updateOutput(input)
   local startTime = sys.clock()
   self:validateParameters()
   input.THNN.Threshold_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.threshold,
      self.val,
      self.inplace
   )
   if self.timerEnable then
                print("Threshold forward time =         ,",self.timeForward," backward time =",self.timeBackward)
                sys.thresholdTime_forward = sys.thresholdTime_forward + self.timeForward
                sys.thresholdTime_backward = sys.thresholdTime_backward + self.timeBackward
                self.timeForward = (sys.clock() - startTime)
                self.cnt = self.cnt + 1
   end
   return self.output
end

function Threshold:updateGradInput(input, gradOutput)
   local startTime = sys.clock()
   self:validateParameters()
   input.THNN.Threshold_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.threshold,
      self.inplace
   )
   if self.timerEnable then
        self.timeBackward = (sys.clock() - startTime)
   end
   return self.gradInput
end

function Threshold:validateParameters()
   self.inplace = self.inplace or false -- backwards compatibility pre inplace
   if self.inplace then
      if self.val > self.threshold then
         error('in-place processing requires value (' .. self.val ..
                  ') not exceed threshold (' .. self.threshold .. ')')
      end
   end
end
