# Durable Medicine Appwrite Schema

`scripts/migrate_durable_medicine_schema.py` declares the non-destructive Appwrite schema used by the durable medicine dispatcher.

It covers `notification_reminders`, `notification_devices`, `med_logs`, and `meds`. The reminder collection has indexed `kind`, `status`, and `sendAtISO` fields for due dispatch. Medicine logs have indexes for occurrence cancellation and time-window seeding. Device and medicine records have `userId` indexes.

Run an inspection (the default) with:

```powershell
python scripts/migrate_durable_medicine_schema.py
```

Only create missing compatible attributes and indexes after an explicit two-part confirmation:

```powershell
python scripts/migrate_durable_medicine_schema.py --apply --confirm-apply
```

The script validates existing types, required flags, sizes, index fields, and index order before it writes. It requires all four collections to exist and never creates collections. It waits for newly created attributes to become available before creating indexes. HTTP 409 is handled as an existing-resource race and is checked for compatibility; it is never reported as an outage. Other request failures are outages. The local journal (`.durable_medicine_schema_journal.json`, configurable with `--journal`) records completed creates so a stopped apply can resume safely. No delete endpoint or delete request is used.

The generated report contains only collection IDs, field names, index names, and statuses. It never prints Appwrite endpoint credentials, project IDs, API keys, device tokens, or document data.

When the durable feature flag is enabled and its infrastructure is unavailable, `/dispatch-due` returns a non-success response with `generic_dispatch_deferred: true`. This deliberately prevents a Scheduler retry from duplicating generic dispatch while durable medicine work is unresolved.
