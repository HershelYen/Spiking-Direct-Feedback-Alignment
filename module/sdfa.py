"""
This file is modified from path 'TinyDFA/tinydfa/dfa.py' in https://github.com/lightonai/dfa-scales-to-modern-deep-learning/
Original author: Julien Launay
Original repository: https://github.com/lightonai/dfa-scales-to-modern-deep-learning/
License: MIT License

Copyright (c) 2020 LightOn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
Modifications by: Haoxiong Ren
Description of modifications: 
- Added remove_indices and keep_indices utility functions.
- Added support for batch dimensions and time dimensions in DFALayer for SNN.
- Added support for spiking direct feedback alignment (SDFA).
- Added support for auto mixing precision training.
"""



# for multi step
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum

# utils
def remove_indices(array, indices):
    # From: https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time
    return [e for i, e in enumerate(array) if i not in set(indices)]

def keep_indices(array, indices):
    return [e for i, e in enumerate(array) if i in set(indices)]

class FeedbackPointsHandling(Enum):
    LAST = 'LAST'
    MINIBATCH = 'MINIBATCH'  # Store all candidate feedback points going through DFALayer (experimental!)
    REDUCE = 'REDUCE'  # Sum feedback points across network and backward on a centralized reduced one

class DFABackend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dfa_context):
        # In minibatch mode, the minibatches have to be tracked:
        if dfa_context.feedback_points_handling == FeedbackPointsHandling.MINIBATCH:
            if dfa_context.forward_complete:
                for layer in dfa_context.dfa_layers:
                    layer.feedback_points = layer.feedback_points[-1:]
                dfa_context.forward_complete = False
                
        ctx.dfa_context = dfa_context
        
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        dfa_context = ctx.dfa_context
        if dfa_context.feedback_points_handling == FeedbackPointsHandling.MINIBATCH:
            # Just backward just started, get the number of minibatches to process:
            # This is to be done even if not training, to free the feedback points eventually.
            if not dfa_context.forward_complete:
                dfa_context.backward_batch = len(dfa_context.dfa_layers[-1].feedback_points)
                dfa_context.forward_complete = True
                
            dfa_context.backward_batch -=1
            this_batch = dfa_context.backward_batch
        # If training, perform the random projection and send it to the feedback points:
        if not dfa_context.no_training:
            grad_size = np.prod(remove_indices(grad_output.shape, dfa_context.batch_dims))
            #print(f"{dfa_context.feedback_matrix.dtype}")
            random_projection = torch.mm(grad_output.reshape(-1, grad_size).to(dfa_context.rp_device),
                                         dfa_context.feedback_matrix.to(dtype=grad_output.dtype))
            #print(f"random_projection_shape{random_projection.shape}")
            if dfa_context.normalization:
                random_projection /= np.sqrt(np.prod(random_projection.shape[1:]))
                
            for layer in dfa_context.dfa_layers:
                # Select the feedback point based on how they are to be handled:
                if dfa_context.feedback_points_handling == FeedbackPointsHandling.MINIBATCH:
                    feedback_point = layer.feedback_points[this_batch]
                elif dfa_context.feedback_points_handling == FeedbackPointsHandling.LAST:
                    feedback_point = layer.feedback_points
                elif dfa_context.feedback_points_handling == FeedbackPointsHandling.REDUCE:
                    feedback_point = dfa_context.global_feedback_point
                feedback_shape = feedback_point.shape[:]
                # remove batch_dims
                size_dim_removed = remove_indices(layer.feedback_shape, layer.batch_dims)
                # remove time_dims -> other_dims
                feedback_size = int(np.prod(remove_indices(size_dim_removed, layer.time_dims)))
                shared_size = np.prod(keep_indices(feedback_point.shape, layer.batch_dims))
                
                if shared_size != random_projection.shape[0]:
                    random_projection_expanded = random_projection.unsqueeze(1)
                    random_projection_expanded = random_projection_expanded.repeat(1, shared_size // random_projection.shape[0], 1)
                    #print("random:",random_projection_expanded.shape)
                    random_projection_expanded /= np.sqrt(np.prod(shared_size // random_projection.shape[0]))
                    random_projection_expanded = random_projection_expanded[:, :, :feedback_size]
                    feedback = random_projection_expanded.view(*feedback_shape).to(feedback_point.device)
                    feedback_point.backward(feedback)
                else:
                    #print(f"random_prj{random_projection.shape}, feedbacK_shape{feedback_shape}, shared_size{shared_size}, feedback_size{feedback_size}")
                    repeat_dims = [feedback_shape[0]] + [1] * (len(feedback_shape)-1)
                    feedback = random_projection[:, :feedback_size].unsqueeze(layer.time_dims[0]).repeat(repeat_dims).reshape(*feedback_shape).to(feedback_point.device)
                    #print(feedback.shape)
                    feedback_point.backward(feedback)

                if dfa_context.feedback_points_handling == FeedbackPointsHandling.REDUCE:
                    # Only need to backward once, on the global feedback point + can free up memory:
                    dfa_context.global_feedback_point = None
                    break

        return grad_output, None  # Gradients for output and dfa_context (None)

class DFA(nn.Module):
    def __init__(self, dfa_layers, normalization=True, rp_device=None, no_training=False,
                 feedback_points_handling=FeedbackPointsHandling.LAST, batch_dims=(0,)):
        super(DFA, self).__init__()
        self.dfa_layers = dfa_layers
        self.normalization = normalization
        self.rp_device = rp_device
        self.no_training = no_training
        self.batch_dims = batch_dims

        # Set the feedback points handling mode of all DFALayers
        self.feedback_points_handling = feedback_points_handling
        for dfa_layer in self.dfa_layers:
            dfa_layer.feedback_registrar = self._register_feedback_point
            dfa_layer.feedback_points_handling = feedback_points_handling

            if dfa_layer.batch_dims is None:
                dfa_layer.batch_dims = self.batch_dims

        self.dfa = DFABackend.apply  # Custom DFA autograd function that actually handles the backward

        # Random feedback matrix and its dimensions
        self.feedback_matrix = None
        self.max_feedback_size = 0
        self.output_size = 0

        # Feedback points handling: minibatch
        self.forward_complete = False
        self.backward_batch = 0

        # Feedback points handling: reduce
        self.global_feedback_point = None

        self.initialized = False

    def forward(self, input):
        if not(self.initialized or self.no_training):
            # If we are training, but aren't initialized:
            # - Setup default rp device if none has been specified;
            # - Get the size of the output (output_size);
            # - Get the size of the largest feedback (max_feedback_size);
            # - Generate the backward random matrix (output_size * max_feedback_size).

            if self.rp_device is None:
                # Default to network output device.
                self.rp_device = input.device
                if self.global_feedback_point is not None:
                    self.global_feedback_point = self.global_feedback_point.to(self.rp_device)
            # input.shape (128, 10) batch_dims = 0
            self.output_size = int(np.prod(remove_indices(input.shape, self.batch_dims)))

            for layer in self.dfa_layers:
                # T, batch_dims, other_dims, checked
                #print(f"layer:{layer.feedback_shape}")
                # remove batch_dims  -> T, other_dims ,checked
                size_dim_removed = remove_indices(layer.feedback_shape, layer.batch_dims)
                # remove time_dims -> other_dims
                feedback_size = int(np.prod(remove_indices(size_dim_removed, layer.time_dims)))
                if feedback_size > self.max_feedback_size:
                    self.max_feedback_size = feedback_size

            #self.feedback_matrix = 2 * (torch.rand(self.output_size, self.max_feedback_size, device=self.rp_device)-0.5)
            self.feedback_matrix = torch.randn(self.output_size, self.max_feedback_size, device=self.rp_device)
            self.initialized = True

        return self.dfa(input, self)

    def _register_feedback_point(self, feedback_point):
        feedback_point_size = np.prod(remove_indices(feedback_point.shape, feedback_point.batch_dims))
        feedback_point = feedback_point.view(-1, feedback_point_size)  # Handle in 1D, put all batch dims together.
        if self.global_feedback_point is None:
            self.global_feedback_point = feedback_point
            if self.rp_device is not None:
                self.global_feedback_point = feedback_point.to(self.rp_device)
        else:
            global_feedback_point_size = np.prod(self.global_feedback_point.shape[1:])
            if global_feedback_point_size > feedback_point_size:
                feedback_point = F.pad(feedback_point.to(self.global_feedback_point.device),
                                       [0, global_feedback_point_size - feedback_point_size])
            elif np.prod(feedback_point.shape) < np.prod(self.global_feedback_point.shape):
                self.global_feedback_point = F.pad(self.global_feedback_point,
                                                   [0, feedback_point_size - global_feedback_point_size])

            self.global_feedback_point = self.global_feedback_point + feedback_point.to(self.global_feedback_point.device)


class DFALayer(nn.Module):
    def __init__(self, name=None, batch_dims=None, time_dims=None, passthrough=False):
        super(DFALayer, self).__init__()

        self.name = name
        self.batch_dims = batch_dims
        self.time_dims = time_dims
        self.passthrough = passthrough

        self.feedback_registrar = None  # Will be specified by topmost DFA layer

        self.feedback_points_handling = None  # Will be specified by topmost DFA layer
        self.feedback_points = None

        self.feedback_shape = None

        self.initialized = False

    def forward(self, input):
        if not self.initialized:
            self.feedback_shape = input.shape
            if self.feedback_points_handling == FeedbackPointsHandling.LAST:
                self.feedback_points = None
            elif self.feedback_points_handling == FeedbackPointsHandling.MINIBATCH:
                self.feedback_points = []

            self.initialized = True

        # Feedback points are useful for backward calculations, only store them if we are calculating gradients:
        if input.requires_grad:  # TODO: input may be a tuple!
            if self.feedback_points_handling == FeedbackPointsHandling.MINIBATCH:
                self.feedback_points.append(input)
            elif self.feedback_points_handling == FeedbackPointsHandling.LAST:
                self.feedback_points = input
            elif self.feedback_points_handling == FeedbackPointsHandling.REDUCE:
                self.feedback_registrar(input)

        # Passthrough mode is used when reproducing the network but training with BP for alignment measurements.
        if self.passthrough:
            return input
        else:
            output = input.detach()  # Cut the computation graph so that gradients don't flow back beyond DFALayer
            output.requires_grad = True  # Gradients will still be required above
            return output
