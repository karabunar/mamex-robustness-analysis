import os
import time
from collections import defaultdict

import numpy as np
import torch

from mamex_robustness.application.diagnostics import positive_ranks
from mamex_robustness.application.evaluation import num_faiss_evaluate
from mamex_robustness.data.loader import PairSample, minibatch
from mamex_robustness.models.criterion import BPR, MSE
from mamex_robustness.tool import cprint, shuffle


class Session(object):
    # Validation noise must not share the test noise mask.
    VAL_NOISE_SEED_OFFSET = 10_000
    # Training noise (--train_noise_rate) is redrawn every epoch from this offset plus the epoch.
    TRAIN_NOISE_SEED_OFFSET = 20_000
    # The gate probe draws a corruption no training epoch used (epochs stay below 10,000).
    PROBE_NOISE_SEED_OFFSET = 30_000

    def __init__(self, env, model, loader):
        self.env = env
        self.model = model
        self.dataset = loader
        self.optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.model.parameters()), 'lr': self.env.args.lr}])
        self.bpr = BPR()
        self.align = MSE()
        self.early_stop = 0
        self.best_epoch = 0
        self.total_epoch = 0
        self.best_ndcg = defaultdict(float)
        self.best_hr = defaultdict(float)
        self.best_recall = defaultdict(float)
        self.test_ndcg = defaultdict(float)
        self.test_hr = defaultdict(float)
        self.test_recall = defaultdict(float)
        self.best_state = None

    def train_epoch(self):
        t = time.time()
        self.model.train()
        self.total_epoch += 1
        S = PairSample(self.dataset)
        users = torch.Tensor(S[:, 0]).long()
        posItems = torch.Tensor(S[:, 1]).long()
        negItems = torch.Tensor(S[:, 2]).long()
        users = users.to(self.env.device)
        posItems = posItems.to(self.env.device)
        negItems = negItems.to(self.env.device)
        users, posItems, negItems = shuffle(users, posItems, negItems)
        total_batch = len(users) // self.env.args.batch_size + 1
        all_loss, all_bpr_loss, all_reg_loss, all_moe_balance_loss, all_fusion_balance_loss = 0., 0., 0., 0., 0.
        all_rbpr_loss, all_align_loss, all_grad_loss = 0., 0., 0.

        for user, pos_item, neg_item, in minibatch(users, posItems, negItems, batch_size=self.env.args.batch_size):
            
            user_emb, item_emb, text_emb, image_emb, moe_balance_loss, fusion_balance_loss = self.model()
            align_loss = self.align(item_emb, text_emb) + self.align(item_emb, image_emb)

            bpr_loss, reg_loss = self.bpr(user_emb, item_emb, user, pos_item, neg_item)
            
            align_loss = self.env.args.align_coeff * align_loss
            reg_loss = self.env.args.reg_coeff * reg_loss
            moe_balance_loss = self.env.args.moe_balance_coeff * moe_balance_loss
            fusion_balance_loss = self.env.args.fusion_balance_coeff * fusion_balance_loss
            loss = bpr_loss + reg_loss + align_loss + moe_balance_loss + fusion_balance_loss 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            all_loss += loss
            all_bpr_loss += bpr_loss
            all_reg_loss += reg_loss
            all_align_loss += align_loss
            all_moe_balance_loss += moe_balance_loss
            all_fusion_balance_loss += fusion_balance_loss
        return all_loss / total_batch, all_bpr_loss / total_batch, all_reg_loss / total_batch, all_align_loss/total_batch, all_moe_balance_loss/total_batch, all_fusion_balance_loss/total_batch, time.time() - t
 
    def train(self, epochs):
        """Train, with VGR off unless --train_with_vgr; the evaluation-time VGR setting is restored afterwards."""
        eval_vgr = self.model.use_vgr
        self.model.use_vgr = bool(getattr(self.env.args, 'train_with_vgr', 0))
        try:
            self._train_loop(epochs)
        finally:
            self.model.use_vgr = eval_vgr

    def _noisy_train_epoch(self, epoch):
        """One training epoch; with --train_noise_rate a fresh share of warm items' text is corrupted for this epoch only."""
        rate = getattr(self.env.args, 'train_noise_rate', 0.0)
        if rate <= 0:
            return self.train_epoch()
        seed = (getattr(self.env.args, 'noise_seed', None) or 0) + self.TRAIN_NOISE_SEED_OFFSET + epoch
        self.model.set_eval_text(self.dataset.train_noisy_text_feat(rate, self.env.args.noise_type, seed))
        try:
            return self.train_epoch()
        finally:
            self.model.set_eval_text(None)

    def _train_loop(self, epochs):
        for epoch in range(self.env.args.ckpt_start_epoch, epochs):
            loss, bpr_loss, reg_loss, align_loss, moe_balance_loss, fusion_balance_loss, train_time = self._noisy_train_epoch(epoch)
            print('-' * 30)
            print(
                f'TRAIN:epoch = {epoch}/{epochs} loss_s1 = {loss:.5f}, bpr_loss_s1 = {bpr_loss:.5f}, align_loss = {align_loss:.5f}, moe_balance_loss = {moe_balance_loss:.5f}, fusion_balance_loss = {fusion_balance_loss:.5f}, train_time = {train_time:.2f}')


            if epoch % self.env.args.eva_interval == 0:
                self.early_stop += self.env.args.eva_interval
                hr, recall, ndcg, val_time = self.test(mode='val', top_list=self.env.args.topk)

                if self.env.args.tensorboard:
                    for key in hr.keys():
                        self.env.w.add_scalar(
                            f'Val/hr@{key}', hr[key], self.total_epoch)
                        self.env.w.add_scalar(
                            f'Val/recall@{key}', hr[key], self.total_epoch)
                        self.env.w.add_scalar(
                            f'Val/ndcg@{key}', ndcg[key], self.total_epoch)
                key = list(hr.keys())[0]
                print(
                    f'epoch = {epoch} hr@{key} = {hr[key]:.5f}, recall@{key} = {recall[key]:.5f}, ndcg@{key} = {ndcg[key]:.5f}, val_time = {val_time:.2f}')

                if ndcg[list(hr.keys())[0]] > self.best_ndcg[list(hr.keys())[0]]:
                    thr, trecall, tndcg, test_time = self.test(mode='test', top_list=self.env.args.topk)
                    self.early_stop = 0
                    for key in thr.keys():
                        cprint(
                            f'epoch = {epoch} hr@{key} = {thr[key]:.5f}, recall@{key} = {trecall[key]:.5f}, ndcg@{key} = {tndcg[key]:.5f}, test_time = {test_time:.2f}')
                    cprint('----------------------')

                    for key in hr.keys():
                        self.best_hr[key] = hr[key]
                        self.best_recall[key] = recall[key]
                        self.best_ndcg[key] = ndcg[key]
                    for key in thr.keys():
                        self.test_hr[key] = thr[key]
                        self.test_recall[key] = trecall[key]
                        self.test_ndcg[key] = tndcg[key]
                    if self.env.args.save:
                        self.save_model(epoch)
                        print('save ckpt')
                    self.best_epoch = epoch
                    self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    if self.env.args.log:
                        self.env.val_logger.info(f'EPOCH[{epoch}/{epochs}]')
                        for key in hr.keys():
                            self.env.val_logger.info(
                                f'hr@{key} = {hr[key]:.5f}, recall@{key} = {recall[key]:.5f}, ndcg@{key} = {ndcg[key]:.5f}, val_time = {val_time:.2f}')

            if self.env.args.log:
                self.env.train_logger.info(
                    f'EPOCH[{epoch}/{epochs}], loss = {loss:.5f}, bpr_loss = {bpr_loss:.5f}, reg_loss = {reg_loss:.5f}')

            if self.env.args.tensorboard:
                self.env.w.add_scalar(f'Train/loss', loss, self.total_epoch)
                self.env.w.add_scalar(
                    f'Train/bpr_loss', bpr_loss, self.total_epoch)
                self.env.w.add_scalar(
                    f'Train/reg_loss', reg_loss, self.total_epoch)

            if self.early_stop > self.env.args.early_stop // 1:
                break


    def _embeddings(self, noisy_text=None, use_vgr=None, vgr_control=None):
        """User/item embeddings (numpy) for one text/VGR configuration; the model's own settings are restored."""
        previous_vgr, previous_control = self.model.use_vgr, self.model.vgr_control
        if use_vgr is not None:
            self.model.use_vgr = bool(use_vgr)
        if vgr_control is not None:
            self.model.vgr_control = vgr_control
        self.model.set_eval_text(noisy_text)
        try:
            user_emb, item_emb, _, _, _, _ = self.model()
        finally:
            self.model.set_eval_text(None)
            self.model.use_vgr = previous_vgr
            self.model.vgr_control = previous_control
        return user_emb.cpu().detach().numpy(), item_emb.cpu().detach().numpy()

    def diagnostics(self, noise_rate, noise_type, noise_seed):
        """Rank of every test positive under each condition, for the failure analysis (model must hold the best state)."""
        self.model.eval()
        noisy = self.dataset.noisy_text_feat(noise_rate, noise_type, noise_seed)
        cold = np.asarray(self.dataset.cold_item_index)
        corrupted = cold[(noisy[cold] != self.dataset.text_feat[cold]).any(axis=1)]
        conditions = {'clean': (None, False, None), 'clean_vgr': (None, True, None), 'noisy': (noisy, False, None), 'noisy_vgr': (noisy, True, None),
                      'noisy_vgr_shuffled': (noisy, True, 'shuffled'), 'noisy_vgr_constant': (noisy, True, 'constant')}
        ranks = {}
        for name, (text, vgr, control) in conditions.items():
            user_emb, item_emb = self._embeddings(text, vgr, control)
            users, items, ranks[name] = positive_ranks(user_emb, item_emb, cold, self.dataset.test_data,
                                                       self.dataset.train_data)
        return {'users': users, 'items': items, 'corrupted_items': corrupted, 'ranks': ranks}

    def test(self, mode='val', top_list=[50], noisy_text=None, use_vgr=None, vgr_control=None):
        """Evaluate on cold items. noisy_text / use_vgr / vgr_control override the model only for this call."""
        self.model.eval()
        t = time.time()
        user_emb, item_emb = self._embeddings(noisy_text, use_vgr, vgr_control)
        data = self.dataset.val_data if mode == 'val' else self.dataset.test_data
        hr, recall, ndcg = num_faiss_evaluate(data,
                                              list(data.keys()),
                                              list(self.dataset.cold_item_index),
                                              self.dataset.train_data,
                                              top_list, user_emb, item_emb)

        return hr, recall, ndcg, time.time() - t

    def _metrics(self, **kwargs):
        hr, recall, ndcg, _ = self.test(mode='test', top_list=self.env.args.topk, **kwargs)
        as_str = lambda d: {str(k): float(v) for k, v in d.items()}
        return {'hr': as_str(hr), 'recall': as_str(recall), 'ndcg': as_str(ndcg)}

    @torch.no_grad()
    def _gate_mean(self, noisy_text, use_vgr, index):
        """Mean [w_v, w_t] over the given items for one text/VGR configuration."""
        previous_vgr = self.model.use_vgr
        self.model.eval()
        self.model.use_vgr = use_vgr
        self.model.set_eval_text(noisy_text)
        try:
            self.model()
        finally:
            self.model.set_eval_text(None)
            self.model.use_vgr = previous_vgr
        w = self.model.last_gate_weights[torch.as_tensor(index, device=self.env.device)].mean(0)
        return {'w_v': float(w[0]), 'w_t': float(w[1])}

    @torch.no_grad()
    def _forward_ms(self, use_vgr, repeats=20):
        previous_vgr = self.model.use_vgr
        self.model.eval()
        self.model.use_vgr = use_vgr
        try:
            self.model()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(repeats):
                self.model()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        finally:
            self.model.use_vgr = previous_vgr
        return (time.perf_counter() - start) / repeats * 1000

    def select_gamma(self, grid, noise_rate, noise_type, noise_seed):
        """Pick the VGR gamma with the best NDCG on the noisy *validation* set; the test set is never used."""
        noisy_val = self.dataset.noisy_text_feat(noise_rate, noise_type, self.VAL_NOISE_SEED_OFFSET + noise_seed)
        key = self.env.args.topk[0]
        previous = self.model.vgr_gamma
        scores = {}
        try:
            for gamma in sorted(float(g) for g in grid):
                self.model.vgr_gamma = gamma
                _, _, ndcg, _ = self.test(mode='val', top_list=self.env.args.topk, noisy_text=noisy_val, use_vgr=True)
                scores[gamma] = float(ndcg[key])
        finally:
            self.model.vgr_gamma = previous
        selected = max(scores, key=lambda g: (scores[g], -g))  # ties go to the smaller gamma
        return {'grid': sorted(scores), 'val_ndcg': {str(g): v for g, v in scores.items()},
                'selected': selected, 'metric': f'ndcg@{key}'}

    def gate_probe(self, noise_rate, noise_type, noise_seed):
        """Mean gate weights on warm items, split by a training-style corruption (model must hold the best state).

        The gate scores one item at a time, so corrupted and uncorrupted items are measured in the same pass.
        """
        noisy = self.dataset.train_noisy_text_feat(noise_rate, noise_type, self.PROBE_NOISE_SEED_OFFSET + noise_seed)
        warm = np.asarray(self.dataset.warm_item_index)
        changed = (noisy[warm] != self.dataset.text_feat[warm]).any(axis=1)
        self.model.eval()
        return {'n_warm_corrupted': int(changed.sum()), 'n_warm_uncorrupted': int((~changed).sum()),
                'warm_clean': self._gate_mean(None, False, warm),
                'warm_corrupted': self._gate_mean(noisy, False, warm[changed]),
                'warm_uncorrupted': self._gate_mean(noisy, False, warm[~changed]),
                'cold_clean': self._gate_mean(None, False, np.asarray(self.dataset.cold_item_index))}

    @torch.no_grad()
    def representation_probe(self):
        """Where the cold items sit relative to the warm items in the space the gate reads (model must hold the best state).

        Clean text, eval mode. The gate input is the concatenation of the image and text expert outputs.
        """
        from scipy.stats import spearmanr
        self.model.eval()
        image_emb, _ = self.model.image_linear(self.model.image_feat)
        text_emb, _ = self.model.text_linear(self.model.text_feat)
        combined = torch.cat([image_emb, text_emb], dim=1)
        logits = self.model.gate(combined)
        weights = torch.softmax(logits, dim=1)
        warm = torch.as_tensor(np.asarray(self.dataset.warm_item_index), device=self.env.device)
        cold = torch.as_tensor(np.asarray(self.dataset.cold_item_index), device=self.env.device)
        distance = (combined - combined[warm].mean(0)).norm(dim=1)

        def group(index):
            return {'norm_v': float(image_emb[index].norm(dim=1).mean()),
                    'norm_t': float(text_emb[index].norm(dim=1).mean()),
                    'distance': float(distance[index].mean()),
                    'logit_gap': float((logits[index, 1] - logits[index, 0]).mean()),
                    'w_t': float(weights[index, 1].mean())}

        warm_stats, cold_stats = group(warm), group(cold)
        rho = spearmanr(distance[cold].cpu().numpy(), weights[cold, 1].cpu().numpy())[0]
        return {'warm': warm_stats, 'cold': cold_stats,
                'distance_ratio': cold_stats['distance'] / warm_stats['distance'],
                'spearman_distance_wt_cold': None if np.isnan(rho) else float(rho)}

    def final_report(self, noise_rate, noise_type, noise_seed, gamma_grid=None):
        """Evaluate the best-validation model on the test set under clean/noisy text, with and without VGR.

        With gamma_grid, gamma is first selected on noisy validation data and the model keeps the selected value.
        """
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.model.eval()
        selection = None
        if gamma_grid:
            selection = self.select_gamma(gamma_grid, noise_rate, noise_type, noise_seed)
            self.model.vgr_gamma = selection['selected']
        noisy = self.dataset.noisy_text_feat(noise_rate, noise_type, noise_seed)
        cold = np.asarray(self.dataset.cold_item_index)
        corrupted = cold[(noisy[cold] != self.dataset.text_feat[cold]).any(axis=1)]
        gate_items = corrupted if len(corrupted) else cold
        vgr = self.model.use_vgr
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        report = {
            'clean': self._metrics(use_vgr=False),
            'clean_vgr': self._metrics(use_vgr=True),
            'noisy': self._metrics(noisy_text=noisy, use_vgr=False),
            'noisy_vgr': self._metrics(noisy_text=noisy, use_vgr=True),
            'noisy_vgr_shuffled': self._metrics(noisy_text=noisy, use_vgr=True, vgr_control='shuffled'),
            'noisy_vgr_constant': self._metrics(noisy_text=noisy, use_vgr=True, vgr_control='constant'),
            'n_corrupted': int(len(corrupted)),
            'gate': {
                'clean': self._gate_mean(None, False, gate_items),
                'noisy': self._gate_mean(noisy, False, gate_items),
                'noisy_vgr': self._gate_mean(noisy, True, gate_items),
            },
            'cost': {
                'forward_ms_plain': self._forward_ms(False),
                'forward_ms_vgr': self._forward_ms(True),
                'gpu_mem_mb': torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0,
            },
        }
        self.model.use_vgr = vgr
        if selection is not None:
            report['gamma_selection'] = selection
        return report

    def save_best(self, path):
        """Write the best-validation weights (kept in memory during training) to path; nothing is written without a best state."""
        if self.best_state is None:
            raise RuntimeError('no best state to save: train first')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.best_state, path)

    def save_ckpt(self, path):
        torch.save(self.model.state_dict(), path)

    def save_model(self, current_epoch):
        model_state_file = os.path.join(
            self.env.CKPT_PATH, f'{self.env.args.suffix}_{self.env.args.penalty_coeff}_epoch{current_epoch}.pth')
        self.save_ckpt(model_state_file)
        if self.best_epoch is not None and current_epoch != self.best_epoch:
            old_model_state_file = os.path.join(
                self.env.CKPT_PATH, f'{self.env.args.suffix}_{self.env.args.penalty_coeff}_epoch{current_epoch}.pth')
            if os.path.exists(old_model_state_file):
                os.system('rm {}'.format(old_model_state_file))
