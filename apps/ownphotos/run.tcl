#!/usr/bin/tclsh

package require Tclx

set secs {1 2 4}
set pids {}

proc start_jobs {} {
    global secs
    global pids
    foreach i $secs {
        set timeout  [expr {$i * 1000}]
        set suffix   "${i}s"
        set childpid [exec ./manage.py consistency --timeout=$timeout --suffix=$suffix > ./log$suffix &]
        lappend pids $childpid
    }
}

proc wait_for_jobs {} {
    global pids
    foreach p $pids {
        wait $p
    }
}

proc kill_jobs {} {
    global pids
    foreach p $pids {
        kill $p
    }
}

proc archive {} {
    set summary_files [glob summary*]
    set export_files [glob export*]
    set log_files [glob log*]
    set timestamp [exec date +%Y%m%d-%H:%M]
    set archive_dir ./archives/$timestamp
    exec mkdir -p $archive_dir
    exec mv [concat $summary_files $export_files $log_files] $archive_dir
}

signal trap SIGINT kill_jobs
start_jobs
wait_for_jobs
archive
