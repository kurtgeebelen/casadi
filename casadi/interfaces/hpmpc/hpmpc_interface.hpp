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


#ifndef CASADI_HPMPC_INTERFACE_HPP
#define CASADI_HPMPC_INTERFACE_HPP

#include "casadi/core/function/conic_impl.hpp"
#include "casadi/core/function/linsol.hpp"
#include <casadi/interfaces/hpmpc/casadi_conic_hpmpc_export.h>

/** \defgroup plugin_Conic_hpmpc
Interface to HMPC Solver


In order to use this interface, you must:

 - Decision variables must only by state and control,
   and the variable ordering must be [x0 u0 x1 u1 ...]
 - The constraints must be in order: [ gap0 lincon0 gap1 lincon1  ]
    
    gap: Ak+1 = Ak xk + Bk uk
    lincon: yk= Ck xk + Dk uk
    
    \verbatim
       A0 B0 -I
       C0 D0
              A1 B1 -I
              C1 D1
    \endverbatim
   
   where I must be a diagonal sparse matrix
 - Either supply all of N, nx, ng, nu options or rely on automatic detection


*/

#ifndef HPMPC_DLOPEN
#include <target.h>
extern "C" {
#include <c_interface.h>
}
#endif

/** \pluginsection{Conic,hpmpc} */

/// \cond INTERNAL
namespace casadi {

  // Forward declaration
  class HpmpcInterface;

  struct CASADI_CONIC_HPMPC_EXPORT HpmpcMemory : public ConicMemory {

    std::vector<double> A, B, b, b2, Q, S, R, q, r, lb, ub, C, D, lg, ug;
    std::vector<double*> As, Bs, bs, Qs, Ss, Rs, qs, rs, lbs, ubs, Cs, Ds, lgs, ugs;

    std::vector<double> I;
    std::vector<double*> Is;
    std::vector<double> x, u, pi, lam;
    std::vector<double*> xs, us, pis, lams;

    std::vector<int> hidxb;
    std::vector<int*> hidxbs;

    std::vector<int> nx;
    std::vector<int> nu;
    std::vector<int> ng;
    std::vector<int> nb;
    std::vector<double> stats;

    std::vector<char> workspace;

    std::vector<double> pv;

    int iter_count;
    int return_status;
    std::vector<double> res;

    /// Constructor
    HpmpcMemory();

    /// Destructor
    ~HpmpcMemory();
  };

  /** \brief \pluginbrief{Conic,hpmpc}
   *
   * @copydoc QPSolver_doc
   * @copydoc plugin_Conic_hpmpc
   *
   * \author Joris Gillis
   * \date 2016
   *
   * */
  class CASADI_CONIC_HPMPC_EXPORT HpmpcInterface : public Conic {
  public:
    /** \brief  Constructor */
    explicit HpmpcInterface();

    /** \brief  Create a new QP Solver */
    static Conic* creator(const std::string& name,
                          const std::map<std::string, Sparsity>& st) {
      return new HpmpcInterface(name, st);
    }

    /** \brief  Create a new Solver */
    explicit HpmpcInterface(const std::string& name,
                              const std::map<std::string, Sparsity>& st);

    /// Get all statistics
    virtual Dict get_stats(void* mem) const;

    /** \brief  Destructor */
    virtual ~HpmpcInterface();

    // Get name of the plugin
    virtual const char* plugin_name() const { return "hpmpc";}

    ///@{
    /** \brief Options */
    static Options options_;
    virtual const Options& get_options() const { return options_;}
    ///@}

    /** \brief  Initialize */
    virtual void init(const Dict& opts);

    /** \brief Create memory block */
    virtual void* alloc_memory() const { return new HpmpcMemory();}

    /** \brief Free memory block */
    virtual void free_memory(void *mem) const { delete static_cast<HpmpcMemory*>(mem);}

    /** \brief Initalize memory block */
    virtual void init_memory(void* mem) const;

    /** \brief  Evaluate numerically */
    virtual void eval(void* mem, const double** arg, double** res, int* iw, double* w) const;

    /// A documentation string
    static const std::string meta_doc;

#ifdef HPMPC_DLOPEN
    /// hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes function
    typedef int (*Work_size)(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, int N2);
    /// fortran_order_d_ip_ocp_hard_tv function
    typedef int (*Ocp_solve)(int *kk, int k_max, double mu0, double mu_tol,
      int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, int N2, int warm_start,
      double **A, double **B, double **b, double **Q, double **S, double **R, double **q,
      double **r, double **lb, double **ub, double **C, double **D, double **lg, double **ug,
      double **x, double **u, double **pi, double **lam, /*double **t,*/
      double *inf_norm_res, void *work0, double *stat);
#endif

  protected:

    struct Block {
      int offset_r;
      int offset_c;
      int rows;
      int cols;
    };

    static Sparsity blocksparsity(int rows, int cols, const std::vector<Block>& b, bool eye=false);
    static void blockptr(std::vector<double *>& vs, std::vector<double>& v,
      const std::vector<Block>& blocks, bool eye=false);
    Sparsity Asp_, Bsp_, Csp_, Dsp_, Isp_, Rsp_, Ssp_, Qsp_, bsp_, lugsp_, usp_, xsp_;
    Sparsity theirs_xsp_, theirs_usp_, theirs_Xsp_, theirs_Usp_;
    Sparsity lamg_gapsp_, lamg_csp_, lam_ulsp_, lam_uusp_, lam_xlsp_, lam_xusp_, lam_clsp_;
    Sparsity lam_cusp_, pisp_;

    std::vector< Block > R_blocks, S_blocks, Q_blocks;
    std::vector< Block > b_blocks, lug_blocks;
    std::vector< Block > u_blocks, x_blocks;
    std::vector< Block > lam_ul_blocks, lam_xl_blocks, lam_uu_blocks, lam_xu_blocks, lam_cl_blocks;
    std::vector< Block > lam_cu_blocks, A_blocks, B_blocks, C_blocks, D_blocks, I_blocks;

    std::vector<int> nxs_;
    std::vector<int> nus_;
    std::vector<int> ngs_;
    int N_;
    int print_level_;

    bool warm_start_;
    double inf_;

    double mu0_; // max element in cost function as estimate of max multiplier
    int max_iter_; // maximum number of iterations
    double tol_; // tolerance in the duality measure

    std::string blasfeo_target_;
    std::string target_;

#ifdef HPMPC_DLOPEN
    Work_size hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes;
    Ocp_solve fortran_order_d_ip_ocp_hard_tv;
#endif


  };

} // namespace casadi

/// \endcond
#endif // CASADI_HPMPC_INTERFACE_HPP
