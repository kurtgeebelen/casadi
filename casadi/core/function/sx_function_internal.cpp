/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            K.U. Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
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


#include "sx_function_internal.hpp"
#include <limits>
#include <stack>
#include <deque>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "../std_vector_tools.hpp"
#include "../sx/sx_node.hpp"
#include "../casadi_types.hpp"
#include "../matrix/sparsity_internal.hpp"
#include "../profiling.hpp"
#include "../casadi_options.hpp"
#include "../casadi_interrupt.hpp"

namespace casadi {

  using namespace std;


  SXFunctionInternal::SXFunctionInternal(const vector<SX >& inputv, const vector<SX >& outputv) :
    XFunctionInternal<SXFunction, SXFunctionInternal, SX, SXNode>(inputv, outputv) {
    setOption("name", "unnamed_sx_function");
    addOption("just_in_time_sparsity", OT_BOOLEAN, false,
              "Propagate sparsity patterns using just-in-time "
              "compilation to a CPU or GPU using OpenCL");
    addOption("just_in_time_opencl", OT_BOOLEAN, false,
              "Just-in-time compilation for numeric evaluation using OpenCL (experimental)");

    casadi_assert(!outputv_.empty()); // NOTE: Remove?

    // Reset OpenCL memory
  }

  SXFunctionInternal::~SXFunctionInternal() {
    // Free OpenCL memory
  }

  void SXFunctionInternal::evalD(const double** arg, double** res,
                                 int* iw, double* w) {
    double time_start=0;
    double time_stop=0;
    if (CasadiOptions::profiling) {
      time_start = getRealTime();
      if (CasadiOptions::profilingBinary) {
        profileWriteEntry(CasadiOptions::profilingLog, this);
      } else {
      CasadiOptions::profilingLog  << "start " << this << ":" <<getOption("name") << std::endl;
      }
    }

    casadi_msg("SXFunctionInternal::evaluate():begin  " << getOption("name"));

    // NOTE: The implementation of this function is very delicate. Small changes in the
    // class structure can cause large performance losses. For this reason,
    // the preprocessor macros are used below
    if (!free_vars_.empty()) {
      std::stringstream ss;
      repr(ss);
      casadi_error("Cannot evaluate \"" << ss.str() << "\" since variables "
                   << free_vars_ << " are free.");
    }


    // Evaluate the algorithm
    for (vector<AlgEl>::iterator it=algorithm_.begin(); it!=algorithm_.end(); ++it) {
      switch (it->op) {
        CASADI_MATH_FUN_BUILTIN(w[it->i1], w[it->i2], w[it->i0])

      case OP_CONST: w[it->i0] = it->d; break;
      case OP_INPUT: w[it->i0] = arg[it->i1]==0 ? 0 : arg[it->i1][it->i2]; break;
      case OP_OUTPUT: if (res[it->i0]!=0) res[it->i0][it->i2] = w[it->i1]; break;
      default:
        casadi_error("SXFunctionInternal::evalD: Unknown operation" << it->op);
      }
    }

    casadi_msg("SXFunctionInternal::evalD():end " << getOption("name"));

    if (CasadiOptions::profiling) {
      time_stop = getRealTime();
      if (CasadiOptions::profilingBinary) {
        profileWriteExit(CasadiOptions::profilingLog, this, time_stop-time_start);
      } else {
        CasadiOptions::profilingLog
          << (time_stop-time_start)*1e6 << " ns | "
          << (time_stop-time_start)*1e3 << " ms | "
          << this << ":" <<getOption("name")
          << ":0||SX algorithm size: " << algorithm_.size()
          << std::endl;
      }
    }
  }


  SX SXFunctionInternal::hess(int iind, int oind) {
    casadi_assert_message(output(oind).numel() == 1, "Function must be scalar");
    SX g = grad(iind, oind);
    g.makeDense();
    if (verbose())  userOut() << "SXFunctionInternal::hess: calculating gradient done " << endl;

    // Create function
    Dict opts;
    opts["verbose"] = getOption("verbose");
    SXFunction gfcn("gfcn", make_vector(inputv_.at(iind)),
                    make_vector(g), opts);

    // Calculate jacobian of gradient
    if (verbose()) {
      userOut() << "SXFunctionInternal::hess: calculating Jacobian " << endl;
    }
    SX ret = gfcn.jac(0, 0, false, true);
    if (verbose()) {
      userOut() << "SXFunctionInternal::hess: calculating Jacobian done" << endl;
    }

    // Return jacobian of the gradient
    return ret;
  }

  bool SXFunctionInternal::isSmooth() const {
    assertInit();

    // Go through all nodes and check if any node is non-smooth
    for (vector<AlgEl>::const_iterator it = algorithm_.begin(); it!=algorithm_.end(); ++it) {
      if (!operation_checker<SmoothChecker>(it->op)) {
        return false;
      }
    }
    return true;
  }

  void SXFunctionInternal::print(ostream &stream) const {
    FunctionInternal::print(stream);

    // Iterator to free variables
    vector<SXElement>::const_iterator p_it = free_vars_.begin();

    // Normal, interpreted output
    for (vector<AlgEl>::const_iterator it = algorithm_.begin(); it!=algorithm_.end(); ++it) {
      InterruptHandler::check();
      if (it->op==OP_OUTPUT) {
        stream << "output[" << it->i0 << "][" << it->i2 << "] = @" << it->i1;
      } else {
        stream << "@" << it->i0 << " = ";
        if (it->op==OP_INPUT) {
          stream << "input[" << it->i1 << "][" << it->i2 << "]";
        } else {
          if (it->op==OP_CONST) {
            stream << it->d;
          } else if (it->op==OP_PARAMETER) {
            stream << *p_it++;
          } else {
            int ndep = casadi_math<double>::ndeps(it->op);
            casadi_math<double>::printPre(it->op, stream);
            for (int c=0; c<ndep; ++c) {
              if (c==0) {
                stream << "@" << it->i1;
              } else {
                casadi_math<double>::printSep(it->op, stream);
                stream << "@" << it->i2;
              }

            }
            casadi_math<double>::printPost(it->op, stream);
          }
        }
      }
      stream << ";" << endl;
    }
  }

  void SXFunctionInternal::generateDeclarations(CodeGenerator& g) const {

    // Make sure that there are no free variables
    if (!free_vars_.empty()) {
      casadi_error("Code generation is not possible since variables "
                   << free_vars_ << " are free.");
    }
  }

  void SXFunctionInternal::generateBody(CodeGenerator& g) const {

    // Which variables have been declared
    vector<bool> declared(sz_w(), false);

    // Run the algorithm
    for (vector<AlgEl>::const_iterator it = algorithm_.begin(); it!=algorithm_.end(); ++it) {
      // Indent
      g.body << "  ";

      if (it->op==OP_OUTPUT) {
        g.body << "if (res[" << it->i0 << "]!=0) "
                      << "res["<< it->i0 << "][" << it->i2 << "]=" << "a" << it->i1;
      } else {
        // Declare result if not already declared
        if (!declared[it->i0]) {
          g.body << "real_t ";
          declared[it->i0]=true;
        }

        // Where to store the result
        g.body << "a" << it->i0 << "=";

        // What to store
        if (it->op==OP_CONST) {
          g.body << g.constant(it->d);
        } else if (it->op==OP_INPUT) {
          g.body << "arg[" << it->i1 << "] ? arg[" << it->i1 << "][" << it->i2 << "] : 0";
        } else {
          int ndep = casadi_math<double>::ndeps(it->op);
          casadi_math<double>::printPre(it->op, g.body);
          for (int c=0; c<ndep; ++c) {
            if (c==0) {
              g.body << "a" << it->i1;
            } else {
              casadi_math<double>::printSep(it->op, g.body);
              g.body << "a" << it->i2;
            }
          }
          casadi_math<double>::printPost(it->op, g.body);
        }
      }
      g.body  << ";" << endl;
    }
  }

  void SXFunctionInternal::init() {

    // Call the init function of the base class
    XFunctionInternal<SXFunction, SXFunctionInternal, SX, SXNode>::init();

    // Stack used to sort the computational graph
    stack<SXNode*> s;

    // All nodes
    vector<SXNode*> nodes;

    // Add the list of nodes
    int ind=0;
    for (vector<SX >::iterator it = outputv_.begin(); it != outputv_.end(); ++it, ++ind) {
      int nz=0;
      for (vector<SXElement>::iterator itc = it->begin(); itc != it->end(); ++itc, ++nz) {
        // Add outputs to the list
        s.push(itc->get());
        sort_depth_first(s, nodes);

        // A null pointer means an output instruction
        nodes.push_back(static_cast<SXNode*>(0));
      }
    }

    // Set the temporary variables to be the corresponding place in the sorted graph
    for (int i=0; i<nodes.size(); ++i) {
      if (nodes[i]) {
        nodes[i]->temp = i;
      }
    }

    // Sort the nodes by type
    constants_.clear();
    operations_.clear();
    for (vector<SXNode*>::iterator it = nodes.begin(); it != nodes.end(); ++it) {
      SXNode* t = *it;
      if (t) {
        if (t->isConstant())
          constants_.push_back(SXElement::create(t));
        else if (!t->isSymbolic())
          operations_.push_back(SXElement::create(t));
      }
    }

    // Use live variables?
    bool live_variables = getOption("live_variables");

    // Input instructions
    vector<pair<int, SXNode*> > symb_loc;

    // Current output and nonzero, start with the first one
    int curr_oind, curr_nz=0;
    for (curr_oind=0; curr_oind<outputv_.size(); ++curr_oind) {
      if (outputv_[curr_oind].nnz()!=0) {
        break;
      }
    }

    // Count the number of times each node is used
    vector<int> refcount(nodes.size(), 0);

    // Get the sequence of instructions for the virtual machine
    algorithm_.resize(0);
    algorithm_.reserve(nodes.size());
    for (vector<SXNode*>::iterator it=nodes.begin(); it!=nodes.end(); ++it) {
      // Current node
      SXNode* n = *it;

      // New element in the algorithm
      AlgEl ae;

      // Get operation
      ae.op = n==0 ? OP_OUTPUT : n->getOp();

      // Get instruction
      switch (ae.op) {
      case OP_CONST: // constant
        ae.d = n->getValue();
        ae.i0 = n->temp;
        break;
      case OP_PARAMETER: // a parameter or input
        symb_loc.push_back(make_pair(algorithm_.size(), n));
        ae.i0 = n->temp;
        break;
      case OP_OUTPUT: // output instruction
        ae.i0 = curr_oind;
        ae.i1 = outputv_[curr_oind].at(curr_nz)->temp;
        ae.i2 = curr_nz;

        // Go to the next nonzero
        curr_nz++;
        if (curr_nz>=outputv_[curr_oind].nnz()) {
          curr_nz=0;
          curr_oind++;
          for (; curr_oind<outputv_.size(); ++curr_oind) {
            if (outputv_[curr_oind].nnz()!=0) {
              break;
            }
          }
        }
        break;
      default:       // Unary or binary operation
        ae.i0 = n->temp;
        ae.i1 = n->dep(0).get()->temp;
        ae.i2 = n->dep(1).get()->temp;
      }

      // Number of dependencies
      int ndeps = casadi_math<double>::ndeps(ae.op);

      // Increase count of dependencies
      for (int c=0; c<ndeps; ++c) {
        refcount.at(c==0 ? ae.i1 : ae.i2)++;
      }
      // Add to algorithm
      algorithm_.push_back(ae);
    }

    // Place in the work vector for each of the nodes in the tree (overwrites the reference counter)
    vector<int> place(nodes.size());

    // Stack with unused elements in the work vector
    stack<int> unused;

    // Work vector size
    size_t worksize = 0;

    // Find a place in the work vector for the operation
    for (vector<AlgEl>::iterator it=algorithm_.begin(); it!=algorithm_.end(); ++it) {

      // Number of dependencies
      int ndeps = casadi_math<double>::ndeps(it->op);

      // decrease reference count of children
      // reverse order so that the first argument will end up at the top of the stack
      for (int c=ndeps-1; c>=0; --c) {
        int ch_ind = c==0 ? it->i1 : it->i2;
        int remaining = --refcount.at(ch_ind);
        if (remaining==0) unused.push(place[ch_ind]);
      }

      // Find a place to store the variable
      if (it->op!=OP_OUTPUT) {
        if (live_variables && !unused.empty()) {
          // Try to reuse a variable from the stack if possible (last in, first out)
          it->i0 = place[it->i0] = unused.top();
          unused.pop();
        } else {
          // Allocate a new variable
          it->i0 = place[it->i0] = worksize++;
        }
      }

      // Save the location of the children
      for (int c=0; c<ndeps; ++c) {
        if (c==0) {
          it->i1 = place[it->i1];
        } else {
          it->i2 = place[it->i2];
        }
      }

      // If binary, make sure that the second argument is the same as the first one
      // (in order to treat all operations as binary) NOTE: ugly
      if (ndeps==1 && it->op!=OP_OUTPUT) {
        it->i2 = it->i1;
      }
    }

    if (verbose()) {
      if (live_variables) {
        userOut() << "Using live variables: work array is "
             <<  worksize << " instead of " << nodes.size() << endl;
      } else {
        userOut() << "Live variables disabled." << endl;
      }
    }

    // Allocate work vectors (symbolic/numeric)
    alloc_w(worksize);
    alloc();
    s_work_.resize(worksize);

    // Reset the temporary variables
    for (int i=0; i<nodes.size(); ++i) {
      if (nodes[i]) {
        nodes[i]->temp = 0;
      }
    }

    // Now mark each input's place in the algorithm
    for (vector<pair<int, SXNode*> >::const_iterator it=symb_loc.begin();
         it!=symb_loc.end(); ++it) {
      it->second->temp = it->first+1;
    }

    // Add input instructions
    for (int ind=0; ind<inputv_.size(); ++ind) {
      int nz=0;
      for (vector<SXElement>::iterator itc = inputv_[ind].begin();
          itc != inputv_[ind].end();
          ++itc, ++nz) {
        int i = itc->getTemp()-1;
        if (i>=0) {
          // Mark as input
          algorithm_[i].op = OP_INPUT;

          // Location of the input
          algorithm_[i].i1 = ind;
          algorithm_[i].i2 = nz;

          // Mark input as read
          itc->setTemp(0);
        }
      }
    }

    // Locate free variables
    free_vars_.clear();
    for (vector<pair<int, SXNode*> >::const_iterator it=symb_loc.begin();
         it!=symb_loc.end(); ++it) {
      if (it->second->temp!=0) {
        // Save to list of free parameters
        free_vars_.push_back(SXElement::create(it->second));

        // Remove marker
        it->second->temp=0;
      }
    }

    // Initialize just-in-time compilation for numeric evaluation using OpenCL
    just_in_time_opencl_ = getOption("just_in_time_opencl");
    if (just_in_time_opencl_) {
      casadi_error("Option \"just_in_time_opencl\" true requires CasADi "
                   "to have been compiled with WITH_OPENCL=ON");
    }

    // Initialize just-in-time compilation for sparsity propagation using OpenCL
    just_in_time_sparsity_ = getOption("just_in_time_sparsity");
    if (just_in_time_sparsity_) {
      casadi_error("Option \"just_in_time_sparsity\" true requires CasADi to "
                   "have been compiled with WITH_OPENCL=ON");
    }

    if (CasadiOptions::profiling && CasadiOptions::profilingBinary) {

      profileWriteName(CasadiOptions::profilingLog, this, getOption("name"),
                       ProfilingData_FunctionType_SXFunction, algorithm_.size());
      int alg_counter = 0;

      // Iterator to free variables
      vector<SXElement>::const_iterator p_it = free_vars_.begin();

      std::stringstream stream;
      for (vector<AlgEl>::const_iterator it = algorithm_.begin(); it!=algorithm_.end(); ++it) {
        stream.str("");
        if (it->op==OP_OUTPUT) {
          stream << "output[" << it->i0 << "][" << it->i2 << "] = @" << it->i1;
        } else {
          stream << "@" << it->i0 << " = ";
          if (it->op==OP_INPUT) {
            stream << "input[" << it->i1 << "][" << it->i2 << "]";
          } else {
            if (it->op==OP_CONST) {
              stream << it->d;
            } else if (it->op==OP_PARAMETER) {
              stream << *p_it++;
            } else {
              int ndep = casadi_math<double>::ndeps(it->op);
              casadi_math<double>::printPre(it->op, stream);
              for (int c=0; c<ndep; ++c) {
                if (c==0) {
                  stream << "@" << it->i1;
                } else {
                  casadi_math<double>::printSep(it->op, stream);
                  stream << "@" << it->i2;
                }

              }
              casadi_math<double>::printPost(it->op, stream);
            }
          }
        }
        stream << std::endl;
        profileWriteSourceLine(CasadiOptions::profilingLog, this,
                               alg_counter++, stream.str(), it->op);
      }
    }

    // Print
    if (verbose()) {
      userOut() << "SXFunctionInternal::init Initialized " << getOption("name") << " ("
           << algorithm_.size() << " elementary operations)" << endl;
    }
  }

  void SXFunctionInternal::evalSX(const SXElement** arg, SXElement** res,
                                  int* iw, SXElement* w) {
    if (verbose()) userOut() << "SXFunctionInternal::evalSXsparse begin" << endl;

    // Iterator to the binary operations
    vector<SXElement>::const_iterator b_it=operations_.begin();

    // Iterator to stack of constants
    vector<SXElement>::const_iterator c_it = constants_.begin();

    // Iterator to free variables
    vector<SXElement>::const_iterator p_it = free_vars_.begin();

    // Evaluate algorithm
    if (verbose()) {
      userOut() << "SXFunctionInternal::evalSXsparse evaluating algorithm forward" << endl;
    }
    for (vector<AlgEl>::const_iterator it = algorithm_.begin(); it!=algorithm_.end(); ++it) {
      switch (it->op) {
      case OP_INPUT:
        w[it->i0] = arg[it->i1]==0 ? 0 : arg[it->i1][it->i2];
        break;
      case OP_OUTPUT:
        if (res[it->i0]!=0) res[it->i0][it->i2] = w[it->i1];
        break;
      case OP_CONST:
        w[it->i0] = *c_it++;
        break;
      case OP_PARAMETER:
        w[it->i0] = *p_it++; break;
      default:
        {
          // Evaluate the function to a temporary value
          // (as it might overwrite the children in the work vector)
          SXElement f;
          switch (it->op) {
            CASADI_MATH_FUN_BUILTIN(w[it->i1], w[it->i2], f)
          }

          // If this new expression is identical to the expression used
          // to define the algorithm, then reuse
          const int depth = 2; // NOTE: a higher depth could possibly give more savings
          f.assignIfDuplicate(*b_it++, depth);

          // Finally save the function value
          w[it->i0] = f;
        }
      }
    }
    if (verbose()) userOut() << "SXFunctionInternal::evalSX end" << endl;
  }

  void SXFunctionInternal::evalFwd(const vector<vector<SX> >& fseed, vector<vector<SX> >& fsens) {
    if (verbose()) userOut() << "SXFunctionInternal::evalFwd begin" << endl;

    // Number of forward seeds
    int nfwd = fseed.size();
    fsens.resize(nfwd);

    // Quick return if possible
    if (nfwd==0) return;

    // Get the number of inputs and outputs
    int num_in = nIn();
    int num_out = nOut();

    // Make sure matching sparsity of fseed
    bool matching_sparsity = true;
    for (int d=0; d<nfwd; ++d) {
      casadi_assert(fseed[d].size()==num_in);
      for (int i=0; matching_sparsity && i<num_in; ++i)
        matching_sparsity = fseed[d][i].sparsity()==input(i).sparsity();
    }

    // Correct sparsity if needed
    if (!matching_sparsity) {
      vector<vector<SX> > fseed2(fseed);
      for (int d=0; d<nfwd; ++d)
        for (int i=0; i<num_in; ++i)
          if (fseed2[d][i].sparsity()!=input(i).sparsity())
            fseed2[d][i] = project(fseed2[d][i], input(i).sparsity());
      return evalFwd(fseed2, fsens);
    }

    // Allocate results
    for (int d=0; d<nfwd; ++d) {
      fsens[d].resize(num_out);
      for (int i=0; i<fsens[d].size(); ++i)
        if (fsens[d][i].sparsity()!=output(i).sparsity())
          fsens[d][i] = SX::zeros(output(i).sparsity());
    }

    // Iterator to the binary operations
    vector<SXElement>::const_iterator b_it=operations_.begin();

    // Tape
    vector<TapeEl<SXElement> > s_pdwork(operations_.size());
    vector<TapeEl<SXElement> >::iterator it1 = s_pdwork.begin();

    // Evaluate algorithm
    if (verbose()) userOut() << "SXFunctionInternal::evalFwd evaluating algorithm forward" << endl;
    for (vector<AlgEl>::const_iterator it = algorithm_.begin(); it!=algorithm_.end(); ++it) {
      switch (it->op) {
      case OP_INPUT:
      case OP_OUTPUT:
      case OP_CONST:
      case OP_PARAMETER:
        break;
      default:
        {
          const SXElement& f=*b_it++;
          switch (it->op) {
            CASADI_MATH_DER_BUILTIN(f->dep(0), f->dep(1), f, it1++->d)
          }
        }
      }
    }

    // Calculate forward sensitivities
    if (verbose())
      userOut() << "SXFunctionInternal::evalFwd calculating forward derivatives" << endl;
    for (int dir=0; dir<nfwd; ++dir) {
      vector<TapeEl<SXElement> >::const_iterator it2 = s_pdwork.begin();
      for (vector<AlgEl>::const_iterator it = algorithm_.begin(); it!=algorithm_.end(); ++it) {
        switch (it->op) {
        case OP_INPUT:
          s_work_[it->i0] = fseed[dir][it->i1].data()[it->i2]; break;
        case OP_OUTPUT:
          fsens[dir][it->i0].data()[it->i2] = s_work_[it->i1]; break;
        case OP_CONST:
        case OP_PARAMETER:
          s_work_[it->i0] = 0;
          break;
          CASADI_MATH_BINARY_BUILTIN // Binary operation
            s_work_[it->i0] = it2->d[0] * s_work_[it->i1] + it2->d[1] * s_work_[it->i2];it2++;break;
        default: // Unary operation
          s_work_[it->i0] = it2->d[0] * s_work_[it->i1]; it2++;
        }
      }
    }
    if (verbose()) userOut() << "SXFunctionInternal::evalFwd end" << endl;
  }

  void SXFunctionInternal::evalAdj(const vector<vector<SX> >& aseed, vector<vector<SX> >& asens) {
    if (verbose()) userOut() << "SXFunctionInternal::evalAdj begin" << endl;

    // number of adjoint seeds
    int nadj = aseed.size();
    asens.resize(nadj);

    // Quick return if possible
    if (nadj==0) return;

    // Get the number of inputs and outputs
    int num_in = nIn();
    int num_out = nOut();

    // Make sure matching sparsity of fseed
    bool matching_sparsity = true;
    for (int d=0; d<nadj; ++d) {
      casadi_assert(aseed[d].size()==num_out);
      for (int i=0; matching_sparsity && i<num_out; ++i)
        matching_sparsity = aseed[d][i].sparsity()==output(i).sparsity();
    }

    // Correct sparsity if needed
    if (!matching_sparsity) {
      vector<vector<SX> > aseed2(aseed);
      for (int d=0; d<nadj; ++d)
        for (int i=0; i<num_out; ++i)
          if (aseed2[d][i].sparsity()!=output(i).sparsity())
            aseed2[d][i] = project(aseed2[d][i], output(i).sparsity());
      return evalAdj(aseed2, asens);
    }

    // Allocate results if needed
    for (int d=0; d<nadj; ++d) {
      asens[d].resize(num_in);
      for (int i=0; i<asens[d].size(); ++i) {
        if (asens[d][i].sparsity()!=input(i).sparsity()) {
          asens[d][i] = SX::zeros(input(i).sparsity());
        } else {
          fill(asens[d][i].begin(), asens[d][i].end(), 0);
        }
      }
    }

    // Iterator to the binary operations
    vector<SXElement>::const_iterator b_it=operations_.begin();

    // Tape
    vector<TapeEl<SXElement> > s_pdwork(operations_.size());
    vector<TapeEl<SXElement> >::iterator it1 = s_pdwork.begin();

    // Evaluate algorithm
    if (verbose()) userOut() << "SXFunctionInternal::evalFwd evaluating algorithm forward" << endl;
    for (vector<AlgEl>::const_iterator it = algorithm_.begin(); it!=algorithm_.end(); ++it) {
      switch (it->op) {
      case OP_INPUT:
      case OP_OUTPUT:
      case OP_CONST:
      case OP_PARAMETER:
        break;
      default:
        {
          const SXElement& f=*b_it++;
          switch (it->op) {
            CASADI_MATH_DER_BUILTIN(f->dep(0), f->dep(1), f, it1++->d)
          }
        }
      }
    }

    // Calculate adjoint sensitivities
    if (verbose()) userOut() << "SXFunctionInternal::evalAdj calculating adjoint derivatives"
                       << endl;
    fill(s_work_.begin(), s_work_.end(), 0);
    for (int dir=0; dir<nadj; ++dir) {
      vector<TapeEl<SXElement> >::const_reverse_iterator it2 = s_pdwork.rbegin();
      for (vector<AlgEl>::const_reverse_iterator it = algorithm_.rbegin();
           it!=algorithm_.rend(); ++it) {
        SXElement seed;
        switch (it->op) {
        case OP_INPUT:
          asens[dir][it->i1].data()[it->i2] = s_work_[it->i0];
          s_work_[it->i0] = 0;
          break;
        case OP_OUTPUT:
          s_work_[it->i1] += aseed[dir][it->i0].data()[it->i2];
          break;
        case OP_CONST:
        case OP_PARAMETER:
          s_work_[it->i0] = 0;
          break;
          CASADI_MATH_BINARY_BUILTIN // Binary operation
            seed = s_work_[it->i0];
          s_work_[it->i0] = 0;
          s_work_[it->i1] += it2->d[0] * seed;
          s_work_[it->i2] += it2->d[1] * seed;
          it2++;
          break;
        default: // Unary operation
          seed = s_work_[it->i0];
          s_work_[it->i0] = 0;
          s_work_[it->i1] += it2->d[0] * seed;
          it2++;
        }
      }
    }
    if (verbose()) userOut() << "SXFunctionInternal::evalAdj end" << endl;
  }

  SXFunctionInternal* SXFunctionInternal::clone() const {
    return new SXFunctionInternal(*this);
  }


  void SXFunctionInternal::clearSymbolic() {
    inputv_.clear();
    outputv_.clear();
    s_work_.clear();
  }

  void SXFunctionInternal::spInit(bool fwd) {
    // Quick return if just-in-time compilation for
    //  sparsity pattern propagation, no work vector needed

    // We need a work array containing unsigned long rather than doubles.
    // Since the two datatypes have the same size (64 bits)
    // we can save overhead by reusing the double array
    alloc();
    bvec_t *iwork = get_bvec_t(w_tmp_);
    if (!fwd) fill_n(iwork, sz_w(), bvec_t(0));
  }

  void SXFunctionInternal::spFwd(const bvec_t** arg, bvec_t** res,
                                 int* iw, bvec_t* w) {
    // Propagate sparsity forward
    for (vector<AlgEl>::iterator it=algorithm_.begin(); it!=algorithm_.end(); ++it) {
      switch (it->op) {
      case OP_CONST:
      case OP_PARAMETER:
        w[it->i0] = 0; break;
      case OP_INPUT:
        w[it->i0] = arg[it->i1]==0 ? 0 : arg[it->i1][it->i2]; break;
      case OP_OUTPUT:
        if (res[it->i0]!=0) res[it->i0][it->i2] = w[it->i1]; break;
      default: // Unary or binary operation
        w[it->i0] = w[it->i1] | w[it->i2]; break;
      }
    }
  }

  void SXFunctionInternal::spAdj(bvec_t** arg, bvec_t** res,
                                 int* iw, bvec_t* w) {
    fill_n(w, sz_w(), 0);

    // Propagate sparsity backward
    for (vector<AlgEl>::reverse_iterator it=algorithm_.rbegin(); it!=algorithm_.rend(); ++it) {
      // Temp seed
      bvec_t seed;

      // Propagate seeds
      switch (it->op) {
      case OP_CONST:
      case OP_PARAMETER:
        w[it->i0] = 0;
        break;
      case OP_INPUT:
        if (arg[it->i1]!=0) arg[it->i1][it->i2] |= w[it->i0];
        w[it->i0] = 0;
        break;
      case OP_OUTPUT:
        if (res[it->i0]!=0) {
          w[it->i1] |= res[it->i0][it->i2];
          res[it->i0][it->i2] = 0;
        }
        break;
      default: // Unary or binary operation
        seed = w[it->i0];
        w[it->i0] = 0;
        w[it->i1] |= seed;
        w[it->i2] |= seed;
      }
    }
  }

  Function SXFunctionInternal::getFullJacobian() {
    SX J = veccat(outputv_).zz_jacobian(veccat(inputv_));
    return SXFunction(name_ + "_jac", inputv_, make_vector(J));
  }


  void SXFunctionInternal::callForward(const std::vector<SX>& arg, const std::vector<SX>& res,
                                   const std::vector<std::vector<SX> >& fseed,
                                   std::vector<std::vector<SX> >& fsens,
                                   bool always_inline, bool never_inline) {
    casadi_assert_message(!never_inline, "SX expressions do not have call nodes");
    XFunctionInternal<SXFunction, SXFunctionInternal, SX, SXNode>
      ::callForward(arg, res, fseed, fsens, true, false);
  }

  void SXFunctionInternal::callReverse(const std::vector<SX>& arg, const std::vector<SX>& res,
                                 const std::vector<std::vector<SX> >& aseed,
                                 std::vector<std::vector<SX> >& asens,
                                 bool always_inline, bool never_inline) {
    casadi_assert_message(!never_inline, "SX expressions do not have call nodes");
    XFunctionInternal<SXFunction, SXFunctionInternal, SX, SXNode>
      ::callReverse(arg, res, aseed, asens, true, false);
  }

} // namespace casadi
