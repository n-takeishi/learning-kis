function [output, args] = load_output(resultdir, mode, no)

args = loadjson([resultdir, 'args.json']);
load([resultdir, 'output_',mode,sprintf('_%d.mat',no)])
output = struct();
output.y0 = y0;
output.g0 = g0;
output.h0 = h0;
output.y1 = y1;
output.g1 = g1;
output.h1 = h1;
output.phi_b = phi_b;
output.phi_W = phi_W;
output.lam = lam;
output.w = w;
output.z = z;
output.rmses = prediction_rmses;