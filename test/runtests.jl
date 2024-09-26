using Test, FairML

out1, out2, out3, out4 = create_data(5000, 0.3, [3;1;1;1;2], "Linear", 1, 42)
out1[1,1] = 42

@test out1[1,1] == 42