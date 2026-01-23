
// Fast version of PHD paper macro - Defensive against vess files

sigmas 		= "4,6"; 
th 			= "0.15"; 
no 			= "20";  
ro 			= "5";  
ni 			= "5";  
step 		= "3";  
kappa 		= "2"; 
ps          = "0.9"; 
pd 			= "0.9"; 
krad    	= "4"; 							
kc			= "30"; 

maxiter  	= "150";
maxepoch 	= "30";

savemidres  = "false";      

arg_str = getArgument;
if (arg_str=="") exit ("Need argument.");
parts = split(arg_str, "::");
main_folder = parts[0];
output_folder = "";
if (parts.length > 1) {
    output_folder = parts[1];
    if (!endsWith(output_folder, "/")) output_folder = output_folder + "/";
}
if (!endsWith(main_folder, "/")) main_folder = main_folder + "/";

setBatchMode(true);
files = getFileList(main_folder);

for (i=0; i<files.length; i++ ) {
    // Check if output exists (assuming simple mapping, file.tif -> file.tif in output)
    if (output_folder != "" && File.exists(output_folder + files[i])) {
        print("Skipping " + files[i] + " (already done)");
        continue;
    }

    // Only process original .tif files, avoid *_vess.tif
	if (endsWith(files[i], ".tif") && indexOf(files[i], "_vess") < 0) { 
		arg = "select="+main_folder+files[i]+" sigmas="+sigmas+" th="+th+" no="+no+" ro="+ro+" ni="+ni+" step="+step+" kappa="+kappa+" ps="+ps+" pd="+pd+" krad="+krad+" kc="+kc+" maxiter="+maxiter+" maxepoch="+maxepoch+" savemidres="+savemidres;
		print("Processing " + files[i]);
		run("PHD", arg); 
	}
}
print("finished.");
