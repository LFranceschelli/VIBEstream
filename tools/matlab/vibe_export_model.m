function vibe_export_model(fname, method, Train, r, r_LR, LR, HR, F, Q, R, varargin)
%VIBE_EXPORT_MODEL  Save resolution-enhancement operators for VibeStream.
%
%   Call it inside the case loop of the MATLAB scripts, after the Kalman-filter
%   setup (the variables below are the scripts' own):
%
%   Proc_Main_FullKF_DEF.m  (Method I):
%       vibe_export_model('model_kf.mat', 'kf', Train, r, r_LR, LR, HR, F, Q, R, C)
%
%   Proc_Main_EPOD_DEF.m  (Method II, or III with Proc.FlagVarRescaling = 1):
%       if Proc.FlagVarRescaling
%           vibe_export_model('model_lse_vr.mat', 'lse_vr', Train, r, r_LR, LR, HR, F, Q, R, M, Gain)
%       else
%           vibe_export_model('model_lse.mat', 'lse', Train, r, r_LR, LR, HR, F, Q, R, M)
%       end
%
%   Q and R are saved as they are at that point, i.e. AFTER tuneQ / tuneR.
%   LR.Um, LR.Vm, HR.Um, HR.Vm must be the means the script subtracted.
%
%   Then in VibeStream: Resolution enhancement > Train a model... > source
%   "Operators computed elsewhere", with the live ROI / window / step the LR
%   fields were computed with, the direction of the Y axis and the velocity
%   units of LR.mat / HR.mat.  See lib/vibe_import.py for the conversion.

S.vibe_format = 'vibe_hr_model_v1';
S.method  = method;
S.r       = r;
S.r_LR    = r_LR;
S.PhiLR   = Train.PhiLR(:, 1:r_LR);
S.SigmaLR = diag(Train.SigmaLR);          % full spectrum (for the energy)
S.PhiMF   = Train.PhiMF(:, 1:r);
S.SigmaMF = diag(Train.SigmaMF);
S.UmLR = LR.Um;  S.VmLR = LR.Vm;  S.UmMF = HR.Um;  S.VmMF = HR.Vm;
S.XLR  = LR.X;   S.YLR  = LR.Y;   S.XMF  = HR.X;   S.YMF  = HR.Y;
S.F = F;  S.Q = Q;
switch method
    case 'kf'
        S.C = varargin{1};           S.R_kf  = R;       % psi_LR = C x
    case 'lse'
        S.M = varargin{1};           S.R_lse = R;       % A ~ B M  (r_LR x r)
    case 'lse_vr'
        S.M = varargin{1};           S.R_vr  = R;
        S.Gain = varargin{2}(:);
    otherwise
        error('vibe_export_model: method must be kf, lse or lse_vr');
end
S.n_train = Train.Nt;
S.created = datestr(now, 31);

w = whos('S');
if w.bytes > 1.9e9
    save(fname, '-struct', 'S', '-v7.3');
else
    save(fname, '-struct', 'S', '-v7');
end
fprintf('VibeStream operators (%s, r = %d, r_LR = %d) saved to %s\n', method, r, r_LR, fname);
end
