import sys
from tqdm import tqdm

import torch
import torch.optim as optim

from models.mutils import save_model
import utils
import losses
from tb_json_logger import log_value


def train_vae(cfgv, model, dataset):
    print('Training base vae ...')
    trainer = optim.Adam(model.vae_params(), lr=cfgv.lr)

    for it in tqdm(range(cfgv.s_iter, cfgv.s_iter + cfgv.n_iter + 1), disable=None):
        if it % cfgv.cheaplog_every == 0 or it % cfgv.expsvlog_every == 0:
            def tblog(k, v):
                log_value('train_' + k, v, it)
        else:
            tblog = lambda k, v: None

        inputs = dataset.next_batch('train_vae')
        beta = utils.anneal(cfgv.beta, it)
        (z_mu, z_logvar), (z, c), dec_logits = model(inputs.text, q_c='prior', sample_z=1)
        recon_loss = losses.recon_dec(inputs.text, dec_logits)
        kl_loss = losses.kl_gaussianprior(z_mu, z_logvar)
        wae_mmd_loss = losses.wae_mmd_gaussianprior(z, method='full_kernel')
        wae_mmdrf_loss = losses.wae_mmd_gaussianprior(z, method='rf')
        z_regu_losses = {'kl': kl_loss, 'mmd': wae_mmd_loss, 'mmdrf': wae_mmdrf_loss}
        z_regu_loss = z_regu_losses[cfgv.z_regu_loss]
        z_logvar_L1 = z_logvar.abs().sum(1).mean(0)  # L1 in z-dim, mean over mb.
        z_logvar_KL_penalty = losses.kl_gaussian_sharedmu(z_mu, z_logvar)
        loss = recon_loss + beta * z_regu_loss \
               + cfgv.lambda_logvar_L1 * z_logvar_L1 \
               + cfgv.lambda_logvar_KL * z_logvar_KL_penalty

        trainer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.vae_params(), cfgv.clip_grad)
        trainer.step()

        tblog('z_mu_L1', z_mu.data.abs().mean().item())
        tblog('z_logvar', z_logvar.data.mean().item())
        tblog('z_logvar_L1', z_logvar_L1.item())
        tblog('z_logvar_KL_penalty', z_logvar_KL_penalty.item())
        tblog('L_vae', loss.item())
        tblog('L_vae_recon', recon_loss.item())
        tblog('L_vae_kl', kl_loss.item())
        tblog('L_wae_mmd', wae_mmd_loss.item())
        tblog('L_wae_mmdrf', wae_mmdrf_loss.item())
        tblog('beta', beta)

        if it % cfgv.cheaplog_every == 0 or it % cfgv.expsvlog_every == 0:
            tqdm.write(
                'ITER {} TRAINING (phase 1). loss_vae: {:.4f}; loss_recon: {:.4f}; loss_kl: {:.4f}; loss_mmd: {:.4f}; '
                'Grad_norm: {:.4e} '
                    .format(it, loss.item(), recon_loss.item(), kl_loss.item(), wae_mmd_loss.item(), grad_norm))

            log_sent, _, _ = model.generate_sentences(1, sample_mode='categorical')
            tqdm.write('Sample (cat T=1.0): "{}"'.format(dataset.idx2sentence(log_sent.squeeze())))
            sys.stdout.flush()
        if it % cfgv.expsvlog_every == 0 and it > 0:
            save_model(model, cfgv.chkpt_path.format(it))
            # Sample 5k sentences from prior/heldout recon/.. to compute external metrics. sample_kwargs from config 
            # start and end of training: do expensive evals too.
            tier = 3 if it == cfgv.s_iter or it == cfgv.s_iter + cfgv.n_iter else 2
