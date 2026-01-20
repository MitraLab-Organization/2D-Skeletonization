
// Vess Runner - Force save?
// It seems arg parsing might ignore 'saveoutput=true' if the key isn't exactly that.
// The log output showed: "saveoutput=false".
// Maybe param name is "save" or "saveimage"?
// Strings dump had "saveAs".
// But let's look at the log: "saveoutput=false". This looks like a dump of current params.
// So the key IS likely "saveoutput".
// Why did it stay false?
// Maybe case sensitive? Or space issue.
// I'll try "saveoutput=true" with spaces clean.
// Or maybe "save=true"?
// I'll try multiple variants in arg.

sigmas 		= "4,6"; 

arg_str = getArgument;
if (arg_str=="") exit ("Need argument: folder.");
parts = split(arg_str, "::");
main_folder = parts[0];
output_folder = "";
if (parts.length > 1) {
    output_folder = parts[1];
    if (!endsWith(output_folder, "/")) output_folder = output_folder + "/";
}
if (!endsWith(main_folder, "/")) {
	main_folder = main_folder + "/";
}

setBatchMode(true);
files = getFileList(main_folder);

for (i=0; i<files.length; i++ ) {
    // Check if output exists in destination
    if (output_folder != "" && File.exists(output_folder + files[i])) {
         print("Skipping " + files[i] + " (already done in output)");
         continue;
    }
    // Also check if intermediate _vess.tif exists in input folder (if previous run crashed before move)
    if (File.exists(main_folder + files[i] + "_vess.tif")) {
         print("Skipping " + files[i] + " (found intermediate result)");
         continue;
    }

    if (endsWith(files[i], ".tif") && indexOf(files[i], "_vess") < 0) { 
        print("Processing " + files[i]);
        // Try passing boolean as string 'true'
        arg = "select=" + main_folder + files[i] + " sigmas=" + sigmas + " saveoutput=true save=true output=true"; 
        
        run("Vess", arg);
        
        // If it doesn't save to file, maybe it returns an ImagePlus?
        // If batch mode is on, it might not show it.
        // I will check nImages.
        if (nImages > 0) {
            print("Image returned: " + getTitle());
            saveAs("Tiff", main_folder + files[i] + "_vess.tif");
            close();
        } else {
            print("No image returned.");
        }
    }
}
print("finished.");
