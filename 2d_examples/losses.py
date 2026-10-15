import torch

def to_sliced_tensor(batch, slices):
  return batch.unsqueeze(0).expand(slices, *batch.shape).contiguous().view(-1, *batch.shape[1:])
def to_sliced_vector(batch, slices):
    sliced_batch = batch.unsqueeze(0).expand(slices, *batch.shape).contiguous().view(-1)
    return sliced_batch

def get_step_fn(reg_type='none', std_value_max=0.5, std_value_min=0.5, balancing_fac=0.1, slices=1, dtype=torch.double, reg_noise_std=0.0, bsp_std_weighting=False):
  def loss_fn(score_model, batch):
    # Perturb the data.
    t = torch.rand(batch.shape[0], device=batch.device)
    std = (std_value_min * (std_value_max / std_value_min) ** t)
    std_ = std[:, None]
    z = torch.randn_like(batch, device=batch.device)
    perturbed_batch = to_sliced_tensor(batch + std_ * z, slices)
    t = to_sliced_vector(t, slices)
    std_ = to_sliced_tensor(std_, slices)
    z = to_sliced_tensor(z, slices)
    perturbed_batch = torch.tensor(perturbed_batch, requires_grad=True, dtype=dtype)

    # Calculate the DSM loss.
    score = score_model(perturbed_batch, t)
    loss = torch.square(score*std_ + z)
    loss = torch.sum(loss, dim=1).mean(dim=0).view(slices, -1).mean(dim=0)

    # Calculate the regularization term.
    if reg_type == 'none':
      reg = torch.zeros((perturbed_batch.shape[0],))
      total_loss = torch.mean(loss)
      total_loss.backward()
    elif reg_type == 'lqc':
      grad0, = torch.autograd.grad(torch.sum(score[torch.arange(score.shape[0]), 0]), perturbed_batch, create_graph=True)
      grad1, = torch.autograd.grad(torch.sum(score[torch.arange(score.shape[0]), 1]), perturbed_batch, create_graph=True)
      traceJJt_true = torch.sum(torch.square(grad0), dim=1) + torch.sum(torch.square(grad1), dim=1) 
      traceJJ_true = torch.square(grad0[:,0]) + torch.square(grad1[:,1]) + 2*grad0[:,1]*grad1[:,0] 
      reg = (traceJJt_true - traceJJ_true)*(std ** 2)
      total_loss = torch.mean(loss + balancing_fac * reg)
      total_loss.backward()
    elif reg_type == 'lqc_est':
      v = torch.randn_like(perturbed_batch, device=batch.device).sign()
      vs = torch.sum(score*v, dim=1)
      gvs, = torch.autograd.grad(torch.sum(vs), perturbed_batch, create_graph=True)
      gvs_detach = torch.tensor(gvs.detach().clone(), requires_grad=True, dtype=dtype)
      gvs_score = torch.sum(score * gvs_detach, dim=1)
      gvss, = torch.autograd.grad(torch.sum(gvs_score), perturbed_batch, create_graph=True)
      traceJJ = torch.sum(v*gvss, dim=1).view(slices, -1).mean(dim=0)
      traceJJt = torch.sum(torch.square(gvs), dim=1).view(slices, -1).mean(dim=0)
      reg = (traceJJt - traceJJ)*(std ** 2)
      total_loss = torch.mean(loss + balancing_fac * reg)
      total_loss.backward()
      gvs.backward(gvs_detach.grad)
    elif reg_type == 'bsp':
      # Bilinear Symmetry Penalty: penalizes (<u,Jv> - <Ju,v>)^2 for two
      # independent probe vectors u,v. Unbiased for ||J-J^T||_F^2, same
      # target as 'lqc'/'lqc_est'. Both terms are ordinary vector-Jacobian
      # products (one backward pass each, seeded with u and v respectively),
      # not Jacobian-vector products, so no double-backward/detach trick is
      # needed here.
      #
      # If reg_noise_std > 0, J is evaluated at the input plus extra Gaussian noise.
      if reg_noise_std > 0:
        x_reg = perturbed_batch + reg_noise_std * torch.randn_like(perturbed_batch, device=batch.device)
        score_reg = score_model(x_reg, t)
      else:
        x_reg = perturbed_batch
        score_reg = score

      u = torch.randn_like(perturbed_batch, device=batch.device).sign()
      v = torch.randn_like(perturbed_batch, device=batch.device).sign()
      Jt_u, = torch.autograd.grad(score_reg, x_reg, grad_outputs=u, create_graph=True, retain_graph=True)
      u_Jv = torch.sum(Jt_u * v, dim=1)
      Jt_v, = torch.autograd.grad(score_reg, x_reg, grad_outputs=v, create_graph=True, retain_graph=True)
      Ju_v = torch.sum(Jt_v * u, dim=1)
      raw_reg = torch.square(u_Jv - Ju_v).view(slices, -1).mean(dim=0)
      # bsp_std_weighting=True multiplies the penalty by std**2, as in the QCSBM estimators.
      reg = raw_reg * (std ** 2) if bsp_std_weighting else raw_reg
      total_loss = torch.mean(loss + balancing_fac * reg)
      total_loss.backward()
    elif reg_type == 'bsp_jvp':
      # Same target and same probes as 'bsp', computed with Jacobian-vector products
      # (double-backward trick).
      u = torch.randn_like(perturbed_batch, device=batch.device).sign()
      v = torch.randn_like(perturbed_batch, device=batch.device).sign()

      def get_jvp(y, x, probe):
        v_base = torch.ones_like(y).requires_grad_(True)
        grads, = torch.autograd.grad(y, x, grad_outputs=v_base, create_graph=True, retain_graph=True)
        jvp, = torch.autograd.grad(grads, v_base, grad_outputs=probe, create_graph=True, retain_graph=True)
        return jvp

      Ju = get_jvp(score, perturbed_batch, u)
      Jv = get_jvp(score, perturbed_batch, v)
      u_Jv = torch.sum(u * Jv, dim=1)
      Ju_v = torch.sum(Ju * v, dim=1)
      reg = torch.square(u_Jv - Ju_v).view(slices, -1).mean(dim=0) * (std ** 2)
      total_loss = torch.mean(loss + balancing_fac * reg)
      total_loss.backward()
    else:
      raise ValueError("Regularization {} not recognized.".format(reg_type))
      
    return loss, reg
    
  def step_fn(score_model, optimizer, batch, is_score_model=True):
    """Running one step of training or evaluation.
    Args:
      score_model: (nn.Module) A parameterized score model.
      optimizer: (torch.optim) An optimizer function that can update score_model with '.step()' function.
      batch: (tensor) A mini-batch of training data.
    Returns:
      loss: (float) The average loss value across the mini-batch.
    """
    optimizer.zero_grad()
    score_model.train()
    if is_score_model:
      loss, reg = loss_fn(score_model, batch)
    else:
      loss, reg = loss_fn(score_model.score, batch)
    torch.nn.utils.clip_grad_norm_(score_model.parameters(), 1.0)
    optimizer.step()
    return torch.mean(loss), torch.mean(reg)

  return step_fn