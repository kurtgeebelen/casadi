/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010 by Joel Andersson, Moritz Diehl, K.U.Leuven. All rights reserved.
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#include "simulator_internal.hpp"
#include "integrator_internal.hpp"
#include "../stl_vector_tools.hpp"
#include "sx_function.hpp"
#include "../sx/sx_tools.hpp"

INPUTSCHEME(SimulatorInput)

using namespace std;
namespace CasADi{

  
SimulatorInternal::SimulatorInternal(const Integrator& integrator, const FX& output_fcn, const vector<double>& grid) : integrator_(integrator), output_fcn_(output_fcn), grid_(grid){
  setOption("name","unnamed simulator");
}
  
SimulatorInternal::~SimulatorInternal(){
}

void SimulatorInternal::init(){
  // Initialize the integrator
  integrator_.init();
  
  // Generate an output function if there is none (returns the whole state)
  if(output_fcn_.isNull()){
    SXMatrix t = symbolic("t");
    SXMatrix x = symbolic("x",integrator_.input(INTEGRATOR_X0).sparsity());
    SXMatrix xdot = symbolic("xp",integrator_.input(INTEGRATOR_XP0).sparsity());
    SXMatrix p = symbolic("p",integrator_.input(INTEGRATOR_P).sparsity());

    vector<SXMatrix> arg(DAE_NUM_IN);
    arg[DAE_T] = t;
    arg[DAE_Y] = x;
    arg[DAE_P] = p;
    arg[DAE_YDOT] = xdot;

    vector<SXMatrix> out(INTEGRATOR_NUM_OUT);
    out[INTEGRATOR_XF] = x;
    out[INTEGRATOR_XPF] = xdot;

    // Create the output function
    output_fcn_ = SXFunction(arg,out);
  }

  // Initialize the output function
  output_fcn_.init();

  // Allocate inputs
  input_.resize(INTEGRATOR_NUM_IN);
  for(int i=0; i<INTEGRATOR_NUM_IN; ++i){
    input(i) = integrator_.input(i);
  }

  // Allocate outputs
  output_.resize(output_fcn_->output_.size());
  for(int i=0; i<output_.size(); ++i)
    output(i) = Matrix<double>(grid_.size(),output_fcn_.output(i).numel(),0);

  // Call base class method
  FXInternal::init();
  
}

void SimulatorInternal::evaluate(int nfdir, int nadir){
  // Pass the parameters and initial state
  integrator_.setInput(input(INTEGRATOR_XF),INTEGRATOR_XF);
  integrator_.setInput(input(INTEGRATOR_XPF),INTEGRATOR_XPF);
  integrator_.setInput(input(INTEGRATOR_P),INTEGRATOR_P);
    
  // Pass sensitivities if fsens
  if(nfdir>0){
    integrator_.setFwdSeed(fwdSeed(INTEGRATOR_XF),INTEGRATOR_XF);
    integrator_.setFwdSeed(fwdSeed(INTEGRATOR_XPF),INTEGRATOR_XPF);
    integrator_.setFwdSeed(fwdSeed(INTEGRATOR_P),INTEGRATOR_P);
  }
  
  // Reset the integrator_
  integrator_.reset(nfdir>0, nadir>0);
  
  // Advance solution in time
  for(int k=0; k<grid_.size(); ++k){

    // Integrate to the output time
    integrator_.integrate(grid_[k]);
    
    // Pass integrator_ output to the output function
    output_fcn_.setInput(grid_[k],DAE_T);
    output_fcn_.setInput(integrator_.output(INTEGRATOR_XF),DAE_Y);
    output_fcn_.setInput(integrator_.output(INTEGRATOR_XPF),DAE_YDOT);
    output_fcn_.setInput(input(INTEGRATOR_P),DAE_P);

    // Evaluate output function
    output_fcn_.evaluate();
    
    // Save the output of the function
    for(int i=0; i<output_.size(); ++i){
      const vector<double> &res = output_fcn_.output(i).data();
      for(int j=0; j<res.size(); ++j){
        output(i)(k,j) = res[j];
      }
    }
    
/*    if(nfdir>0){
      
      // Pass the forward seed to the output function
      output_fcn_.setFwdSeed(integrator_.fwdSens(DAE_Y));
      output_fcn_.setFwdSeed(fwdSeed(INTEGRATOR_P),DAE_P);
      
      // Evaluate output function
      output_fcn_.evaluate(nfdir,0);

      // Save the output of the function
      for(int i=0; i<output_.size(); ++i){
        const vector<double> &res = output_fcn_.fwdSens(i).data();
        copy(res.begin(),res.end(),&fwdSens(i).at(k*res.size()));
      }
    }*/
  }
  
  // Adjoint sensitivities
  if(nadir>0){

    #if 0
          // Clear the seeds (TODO: change this when XF is included as output of the simulator!)
    vector<double> &xfs = integrator_.output(INTEGRATOR_XF).aseed();
    fill(xfs.begin(),xfs.end(),0);
    vector<double> &x0s = integrator_.input(INTEGRATOR_X0,1);
    fill(x0s.begin(),x0s.end(),0);
    vector<double> &ps = integrator_.input(INTEGRATOR_P,1);
    fill(ps.begin(),ps.end(),0);
    vector<double> &ps_sim = input(SIMULATOR_P).data(1);
    fill(ps_sim.begin(),ps_sim.end(),0);
    
    // Reset the integrator for backward integration
    integrator_->resetAdj();

    // Seeds from the output function
    const vector<double> &xf_seed = output(0).data(1); // TODO: output is here assumed to be trivial, returning the state
    casadi_assert(xf_seed.size() == grid_.size()*xfs.size());

    // Integrate backwards
    for(int k=grid_.size()-1; k>=0; --k){
      // Integrate back to the previous grid point
      integrator_->integrateAdj(grid_[k]);
      
      // Pass adjoint seeds to integrator
      for(int i=0; i<xfs.size(); ++i)
        xfs.at(i) = x0s.at(i) + xf_seed.at(k*xfs.size() + i);
      
      // Add the contribution to the parameter sensitivity
      for(int i=0; i<ps.size(); ++i)
        ps_sim[i] += ps[i];

      // Reset the integrator to deal with the jump in seeds
      integrator_->resetAdj();
    }
    
    // Save
    vector<double> &x0_sim = input(SIMULATOR_X0).data(1);
    copy(x0s.begin(),x0s.end(),x0_sim.begin());

    #endif
    
  }

}

} // namespace CasADi


