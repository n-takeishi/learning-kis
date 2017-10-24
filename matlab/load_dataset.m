function [y0, y1] = load_dataset(datadir, mode, no)

y = importdata([datadir,'/',mode,sprintf('_%d.txt',no)]);

if nargout==2
    y0 = y(1:end-1,:);
    y1 = y(2:end,:);
else
    y0 = y;
end