function plot_adaptation_diagnostics(log, target, n_baseline)
% PLOT_ADAPTATION_DIAGNOSTICS  6-panel post-run diagnostic for one adaptive
% formant-perturbation session.
%
%   log         struct of per-recorded-trial arrays (see logging snippet):
%       .trial  [N x1] trial index
%       .obs    [N x2] measured mid-vowel (F1,F2), raw Hz
%       .act    [N x2] agent action [dF1,dF2] computed THIS trial; this action
%                       is APPLIED on the NEXT trial. NaN on baseline trials.
%       .shift  [N x2] realized shift THIS trial = sfmts-fmts, SIGNED, raw Hz
%       .k      [N x2] RLS diagonal-gain estimate [kF1,kF2] after the update
%       .cross  [N x2] RLS cross-coupling [F1<-F2, F2<-F1] after the update
%       .base   [N x1] logical, true on zero-shift baseline trials
%   target      [F1 F2] agent target, raw Hz
%   n_baseline  number of opening baseline trials (unused except for context)

tr    = log.trial(:);
obs   = log.obs;
act   = log.act;
shift = log.shift;
k     = log.k;
cross = log.cross;
base  = logical(log.base(:));
ctrl  = ~base;                                   % control (post-baseline) trials

dist  = sqrt(sum((obs - target(:)').^2, 2));     % distance to target each trial

F1C = [0.10 0.60 0.20];     % green  = F1
F2C = [0.70 0.20 0.70];     % purple = F2
BLU = [0.20 0.40 0.80];     % blue   = "F1-channel" estimate
ORG = [0.90 0.40 0.10];     % orange = "F2-channel" estimate

figure('Name','Adaptation diagnostics','Color','w','Position',[60 60 1500 860]);
tl = tiledlayout(2,3,'Padding','compact','TileSpacing','compact');

% ---- 1. Convergence to target -----------------------------------------
nexttile; hold on;
xb = tr(base);
if ~isempty(xb)
    xregion(min(xb)-0.5, max(xb)+0.5, 'FaceColor',[0.86 0.86 0.86]);
    text(min(xb)-0.4, max(dist)*0.97, ' baseline', 'FontSize',8, ...
         'VerticalAlignment','top');
end
plot(tr, dist, '-o','Color',BLU,'MarkerFaceColor',BLU,'MarkerSize',4);
ylim([0 max(dist)*1.1]); xlabel('Trial'); ylabel('Distance to target (Hz)');
title('1. Convergence to target'); grid on; box on;

% ---- 2. Production path in F1-F2 space (colored by trial) -------------
nexttile; hold on;
plot(obs(:,2), obs(:,1), '-', 'Color',[0.7 0.7 0.7 0.5]);   % faint thread
scatter(obs(:,2), obs(:,1), 50, tr, 'filled', ...
        'MarkerEdgeColor','k','LineWidth',0.3);
plot(obs(1,2),  obs(1,1),  's','MarkerSize',12, ...
     'MarkerFaceColor',[0.80 0.20 0.20],'MarkerEdgeColor','k');   % start
plot(target(2), target(1), 'p','MarkerSize',17, ...
     'MarkerFaceColor',[0.95 0.80 0.10],'MarkerEdgeColor','k');   % target
set(gca,'XDir','reverse','YDir','reverse');                 % phonetic convention
xlabel('F2 (Hz)'); ylabel('F1 (Hz)');
title('2. Production path  (\bullet=start, \star=target)');
cb = colorbar; cb.Label.String = 'Trial'; colormap(gca, parula); grid on; box on;

% ---- 3. Measured formants vs target -----------------------------------
nexttile; hold on;
plot(tr, obs(:,1), '-o','Color',F1C,'MarkerSize',3);
plot(tr, obs(:,2), '-o','Color',F2C,'MarkerSize',3);
yline(target(1),'LineStyle','--','Color',F1C,'LineWidth',1);
yline(target(2),'LineStyle','--','Color',F2C,'LineWidth',1);
xlabel('Trial'); ylabel('Formant (Hz)');
legend({'F1','F2','F1 target','F2 target'},'Location','best','FontSize',8);
title('3. Measured formants vs target'); grid on; box on;

% ---- 4. RLS gain estimate ---------------------------------------------
nexttile; hold on;
plot(tr(ctrl), k(ctrl,1), '-o','Color',BLU,'MarkerSize',3);
plot(tr(ctrl), k(ctrl,2), '-o','Color',ORG,'MarkerSize',3);
yline(0,'k:');
xlabel('Trial'); ylabel('Estimated gain k');
legend({'k_{F1}','k_{F2}'},'Location','best','FontSize',8);
title('4. RLS gain estimate'); grid on; box on;

% ---- 5. Intended vs realized shift  (SIGNED, same-trial aligned) -------
% AdaptationRun logs log_action(t) as the action APPLIED on trial t (loaded
% into pertAmp/pertPhi at the top of t, computed at the end of t-1), on the
% same row as the obs/sobs it produced. Its realization is therefore
% shift(t) = sobs(t)-obs(t) -- the SAME row, not t+1.
nexttile; hold on;
r  = find(~isnan(act(:,1)));
xr = tr(r);
plot(xr, act(r,1),    '-o','Color',F1C,'MarkerSize',3);
plot(xr, shift(r,1),  '--s','Color',F1C,'MarkerSize',3);
plot(xr, act(r,2),    '-o','Color',F2C,'MarkerSize',3);
plot(xr, shift(r,2),  '--s','Color',F2C,'MarkerSize',3);
yline(0,'k:');
xlabel('Trial (perturbation applied)'); ylabel('Shift (Hz, signed)');

% ---- 6. RLS cross-coupling estimate -----------------------------------
nexttile; hold on;
plot(tr(ctrl), cross(ctrl,1), '-o','Color',BLU,'MarkerSize',3);
plot(tr(ctrl), cross(ctrl,2), '-o','Color',ORG,'MarkerSize',3);
yline(0,'k:');
xlabel('Trial'); ylabel('Estimated cross gain');
legend({'F1\leftarrowF2','F2\leftarrowF1'},'Location','best','FontSize',8);
title('6. RLS cross-coupling estimate'); grid on; box on;

title(tl, sprintf('Adaptation diagnostics   (target F1=%.0f  F2=%.0f Hz)', ...
                  target(1), target(2)), 'FontWeight','bold');
end
