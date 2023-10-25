function data= loadPD(filename)
%Load .mat data
data = load(filename);
data = data.A2;
end