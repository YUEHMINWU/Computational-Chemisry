# Load the PDB and DCD files
mol new start_drudes.pdb
mol addfile FV_NVT.dcd

# Open and read charge file
set chargeFile [open "charges.dat" r]
set charges [read $chargeFile]
close $chargeFile
set chargeData [split $charges "\n"]

# Delete any existing representations
mol delrep all top

# Find the actual min/max charges from the data for better scaling
set min_charge 0
set max_charge 0
foreach line $chargeData {
    set values [split $line " "]
    foreach charge $values {
        if {$charge != ""} {
            if {$charge < $min_charge} {
                set min_charge $charge
            }
            if {$charge > $max_charge} {
                set max_charge $charge
            }
        }
    }
}

# Function to convert charge value to RGB color (BWR scheme)
proc charge_to_color {charge min_charge max_charge} {
    set range [expr {max(abs($min_charge), abs($max_charge))}]
    set normalized [expr {$charge / $range}]
    
    if {$normalized < 0} {
        # Blue to White (negative)
        set intensity [expr {1.0 + $normalized}]
        return [list $intensity $intensity 1.0]
    } else {
        # White to Red (positive)
        set intensity [expr {1.0 - $normalized}]
        return [list 1.0 $intensity $intensity]
    }
}

# Create representations for each chain
# Chain A (positive electrode)
mol representation VDW 1.0
mol selection "chain A"
mol color User
mol addrep top
mol modmaterial 0 top Opaque

# Chain B (negative electrode)
mol representation VDW 1.0
mol selection "chain B"
mol color User
mol addrep top
mol modmaterial 1 top Opaque

# Process each frame
for {set frame 0} {$frame < 10} {incr frame} {
    # Move to current frame
    animate goto $frame
    
    # Get charge data for current frame
    set frameData [lindex $chargeData $frame]
    set chargeList [split $frameData " "]
    
    # Process Chain A (positive electrode)
    set index 0
    set grpA_atoms [atomselect top "chain A"]
    $grpA_atoms frame $frame
    set chainA_charges {}
    foreach atom [$grpA_atoms list] {
        set charge [lindex $chargeList $index]
        # Normalize charge for coloring
        set normalized_charge [expr {($charge - $min_charge) / ($max_charge - $min_charge)}]
        lappend chainA_charges $normalized_charge
        incr index
    }
    $grpA_atoms set user $chainA_charges
    
    # Process Chain B (negative electrode)
    set grpB_atoms [atomselect top "chain B"]
    $grpB_atoms frame $frame
    set chainB_charges {}
    foreach atom [$grpB_atoms list] {
        set charge [lindex $chargeList $index]
        # Normalize charge for coloring
        set normalized_charge [expr {($charge - $min_charge) / ($max_charge - $min_charge)}]
        lappend chainB_charges $normalized_charge
        incr index
    }
    $grpB_atoms set user $chainB_charges
    
    # Clean up selections
    $grpA_atoms delete
    $grpB_atoms delete
    
    # Update display
    display update
}

# Create color scale window
proc create_colorbar {min_charge max_charge} {
    # Destroy existing color bar if it exists
    if {[winfo exists ".colorbar"]} {
        destroy ".colorbar"
    }
    
    # Create color bar window
    set colorbar [toplevel ".colorbar"]
    wm title $colorbar "Charge Distribution (e)"
    
    # Create canvas for the color bar
    canvas $colorbar.canvas -width 300 -height 100
    pack $colorbar.canvas

    # Create the gradient
    for {set i 0} {$i < 300} {incr i} {
        # Calculate charge value for this position
        set fraction [expr {$i / 300.0}]
        set charge [expr {$min_charge + ($max_charge - $min_charge) * $fraction}]
        
        # Get RGB colors using the same function as the main visualization
        lassign [charge_to_color $charge $min_charge $max_charge] red green blue
        
        # Convert RGB values to hex color
        set color [format "#%02x%02x%02x" \
            [expr {int($red * 255)}] \
            [expr {int($green * 255)}] \
            [expr {int($blue * 255)}]]
        
        # Draw vertical line of the gradient
        $colorbar.canvas create line $i 20 $i 50 -fill $color
    }
    
    # Add tick marks and labels
    set num_ticks 5
    set tick_interval [expr {300.0 / ($num_ticks - 1)}]
    set charge_interval [expr {($max_charge - $min_charge) / ($num_ticks - 1)}]
    
    for {set i 0} {$i < $num_ticks} {incr i} {
        set x [expr {$i * $tick_interval}]
        set charge [expr {$min_charge + $i * $charge_interval}]
        
        # Draw tick mark
        $colorbar.canvas create line $x 50 $x 60 -width 2
        
        # Add label
        $colorbar.canvas create text $x 70 -anchor c \
            -text [format "%.3f" $charge]
    }
}

# Set final coloring and materials
mol modcolor 0 top User
mol modmaterial 0 top Opaque
mol modcolor 1 top User
mol modmaterial 1 top Opaque

# Center and reset view
display resetview
display update

# Create the color bar
create_colorbar $min_charge $max_charge

puts "Visualization completed with charge range: $min_charge to $max_charge"
