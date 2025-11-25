import torch
import torch.nn.functional as F


def hessian_power_iteration(model, loss, num_iters=20, tol=1e-5):
    """
    Memory-optimized estimation of largest eigenvalue of Hessian.
    
    Key optimizations:
    - CPU offloading for intermediate vectors
    - Gradient checkpointing compatibility
    - Efficient memory reuse
    - No retained computation graphs between iterations
    """
    model.zero_grad()
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Initialize random vector on CPU to save GPU memory
    total_params = sum(p.numel() for p in params)
    v_cpu = torch.randn(total_params, dtype=torch.float32)
    v_cpu = v_cpu / v_cpu.norm()
    
    prev_lambda = 0.0
    
    for i in range(num_iters):
        # Move v to GPU only when needed
        v = v_cpu.to(loss.device)
        
        # Compute Hessian-vector product efficiently
        hv = hessian_vector_product_optimized(model, loss, params, v)
        
        # Compute eigenvalue estimate
        lambda_est = torch.dot(v, hv).item()
        
        # Normalize and move back to CPU
        hv_norm = hv.norm().item()
        v_cpu = (hv / (hv_norm + 1e-12)).cpu()
        
        # Clean up GPU memory
        del v, hv
        torch.cuda.empty_cache()
        
        # Check convergence
        if abs(lambda_est - prev_lambda) < tol:
            print(f"Converged at iteration {i+1}")
            break
        prev_lambda = lambda_est
    
    return lambda_est, v_cpu


def hessian_vector_product_optimized(model, loss, params, v_flat):
    """
    Compute Hessian-vector product with minimal memory footprint.
    Uses the R-operator (forward-mode autodiff) approach.
    """
    # Split vector into per-parameter chunks
    v_parts = []
    idx = 0
    for p in params:
        numel = p.numel()
        v_parts.append(v_flat[idx:idx + numel].view_as(p).detach())
        idx += numel
    
    # First backward pass: compute gradients
    model.zero_grad()
    grads = torch.autograd.grad(
        loss, params, create_graph=True, retain_graph=False
    )
    
    # Compute grad^T @ v (scalar)
    grad_v = sum(
        (g * v).sum() 
        for g, v in zip(grads, v_parts)
    )
    
    # Second backward pass: compute H @ v
    model.zero_grad()
    hv_parts = torch.autograd.grad(
        grad_v, params, retain_graph=False
    )
    
    # Flatten and detach immediately
    hv_flat = torch.cat([
        h.detach().contiguous().view(-1) for h in hv_parts
    ])
    
    return hv_flat


# def hessian_power_iteration_ultra_low_memory(model, trainer, batch, num_iters=20, tol=1e-5, num_batches=1):
#     """
#     Ultra low memory version with multiple batches.
    
#     Args:
#         loss_fn: Function that takes (model, batch) and returns loss
#         dataloader: Dataloader (will use first num_batches batches)
#         num_batches: Number of batches to average over
#     """
#     params = [p for p in model.parameters() if p.requires_grad]
#     total_params = sum(p.numel() for p in params)
    
#     # Keep v on CPU
#     v_cpu = torch.randn(total_params, dtype=torch.float32)
#     v_cpu = v_cpu / v_cpu.norm()
    
#     # Determine device from model parameters
#     device = next(model.parameters()).device
    
#     prev_lambda = 0.0
    
#     for i in range(num_iters):
#         v = v_cpu.to(device)  # Changed from .cuda() to .to(device)
        
#         # Accumulate Hessian-vector product across multiple batches
#         # hv_flat_accumulated = torch.zeros_like(v)
        
#         # for batch_idx, batch in enumerate(dataloader):
#         #     if batch_idx >= num_batches:
#         #         break
                
#             # Recompute loss for this batch
#         model.zero_grad()
#         logits = model(batch[0])
#         # loss = trainer.compute_loss(model, batch, False)
#         loss_fn = torch.nn.CrossEntropyLoss()
#         loss = loss_fn(logits, batch[1])
            
#         # First order gradients
#         grads = torch.autograd.grad(loss, params, create_graph=True)
            
#         # Split v and ensure it's on the correct device
#         v_parts = []
#         idx = 0
#         for p in params:
#             numel = p.numel()
#             v_part = v[idx:idx + numel].view_as(p).to(p.device)  # Ensure same device as param
#             v_parts.append(v_part)
#             idx += numel
            
#         # grad^T @ v
#         grad_v = sum((g * vp).sum() for g, vp in zip(grads, v_parts))
            
#         # H @ v for this batch
#         model.zero_grad()
#         hv_parts = torch.autograd.grad(grad_v, params)
#         hv_flat = torch.cat([h.detach().view(-1).to(device) for h in hv_parts])  # Ensure all on same device
            
#             # # Accumulate
#         # hv_flat_accumulated = hv_flat / num_batches
            
#             # Cleanup per batch
#         # del grads, grad_v, loss, v_parts
#         # torch.cuda.empty_cache()
        
#         # Update with averaged Hessian-vector product
#         lambda_est = torch.dot(v, hv_flat).item()
#         hv_norm = hv_flat.norm().item()
#         v_cpu = (hv_flat / (hv_norm + 1e-12)).cpu()
        
#         # Aggressive cleanup
#         del v, hv_flat
#         torch.cuda.empty_cache()
        
#         if abs(lambda_est - prev_lambda) < tol:
#             print(f"Converged at iteration {i+1}")
#             break
#         prev_lambda = lambda_est
    
#     return lambda_est, v_cpu



def hessian_power_iteration_ultra_low_memory(model, trainer, batch, num_iters=5, tol=1e-5, num_batches=1):
    """
    Ultra low memory version using proper power iteration for largest singular value.
    Based on standard power iteration: alternates between v = Gu/||Gu|| and u = G*v/||G*v||
    
    Args:
        model: PyTorch model
        trainer: Trainer object (not used in current implementation)
        batch: Single batch of data (tuple of inputs and labels)
        num_iters: Number of power iterations
        tol: Convergence tolerance
        num_batches: Number of batches (not used in current implementation)
    """
    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params)
    
    # Initialize random vector u on CPU
    u_cpu = torch.randn(total_params, dtype=torch.float32)
    u_cpu = u_cpu / u_cpu.norm()
    
    # Determine device from model parameters
    device = next(model.parameters()).device
    
    prev_sigma = 0.0
    
    for i in range(num_iters):
        # Step 1: v = Gu / ||Gu||  where G is the Hessian
        u = u_cpu.to(device)
        
        # Compute Hessian @ u (this is Gu)
        model.zero_grad()
        loss = trainer.compute_loss(model, batch, False)
        # logits = model(batch[0])
        # loss_fn = torch.nn.CrossEntropyLoss()
        # loss = loss_fn(logits, batch[1])
        
        # First order gradients
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        # Split u into parameter shapes
        u_parts = []
        idx = 0
        for p in params:
            numel = p.numel()
            u_part = u[idx:idx + numel].view_as(p).to(p.device)
            u_parts.append(u_part)
            idx += numel
        
        # grad^T @ u
        grad_u = sum((g * up).sum() for g, up in zip(grads, u_parts))
        
        # Gu = H @ u
        model.zero_grad()
        gu_parts = torch.autograd.grad(grad_u, params)
        gu_flat = torch.cat([h.detach().view(-1).to(device) for h in gu_parts])
        
        # Normalize: v = Gu / ||Gu||
        gu_norm = gu_flat.norm().item()
        if gu_norm < 1e-12:
            print(f"Warning: Gu norm too small at iteration {i+1}")
            break
        v = gu_flat / gu_norm
        
        # Cleanup before step 2
        del grads, grad_u, u_parts, gu_parts, gu_flat, u
        torch.cuda.empty_cache()
        
        # Step 2: u = G* @ v / ||G* @ v||  where G* is G^T (Hessian is symmetric so G* = G)
        model.zero_grad()
        # logits = model(batch[0])
        # loss = loss_fn(logits, batch[1])
        loss = trainer.compute_loss(model, batch, False)

        
        # First order gradients
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        # Split v into parameter shapes
        v_parts = []
        idx = 0
        for p in params:
            numel = p.numel()
            v_part = v[idx:idx + numel].view_as(p).to(p.device)
            v_parts.append(v_part)
            idx += numel
        
        # grad^T @ v
        grad_v = sum((g * vp).sum() for g, vp in zip(grads, v_parts))
        
        # G* @ v = H @ v (Hessian is symmetric)
        model.zero_grad()
        gv_parts = torch.autograd.grad(grad_v, params)
        gv_flat = torch.cat([h.detach().view(-1).to(device) for h in gv_parts])
        
        # Normalize: u = G*v / ||G*v||
        gv_norm = gv_flat.norm().item()
        if gv_norm < 1e-12:
            print(f"Warning: G*v norm too small at iteration {i+1}")
            break
        u_cpu = (gv_flat / gv_norm).cpu()
        
        # Estimate singular value: σ = (Gu)^T v
        # We already have v and gu_norm from step 1
        # Since v = Gu/||Gu||, we have (Gu)^T v = ||Gu|| * v^T v = ||Gu||
        sigma_est = gu_norm
        
        # Cleanup
        del v, grads, grad_v, v_parts, gv_parts, gv_flat
        torch.cuda.empty_cache()
        
        # Check convergence
        if abs(sigma_est - prev_sigma) < tol:
            print(f"Converged at iteration {i+1}, σ₁ = {sigma_est:.6f}")
            break
        prev_sigma = sigma_est
        
        # if (i + 1) % 5 == 0:
        #     print(f"Iteration {i+1}/{num_iters}, σ₁ ≈ {sigma_est:.6f}")
    
    return sigma_est, loss.item()