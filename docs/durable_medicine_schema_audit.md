# Durable Medicine Schema Audit

Read-only Appwrite audit performed with the migration script's `--audit` and default dry-run modes. No schema, scheduler, flag, or deployment changes were made.

## Current Compatibility

| Collection | Present durable fields/indexes | Missing additions |
| --- | --- | --- |
| `notification_reminders` | Shared generic fields, `userId,eventId` index | `occurrenceId`, `kind`, claim/dispatch timestamps, attempt count; due/event/key/occurrence indexes |
| `med_logs` | User, medicine, status, datetime time fields; single-field time/status indexes | `occurrenceId`; occurrence and bounded dose composite indexes |
| `meds` | Required medicine fields | `userId` index for non-scan lookup |
| `notification_devices` | Required registration fields and user index | Named durable user index is redundant with the existing user index |

## Queries Requiring Indexes

- Due reminders: `kind`, `status`, `sendAtISO` ascending.
- Reminder identity: `userId,eventId`; `notificationKey`; `userId,occurrenceId`.
- Occurrence seed window: `med_logs.time`.
- Exact/legacy dose matching: `userId,occurrenceId`; `userId,medId,time`; `userId,medId,status,time`.
- Medicine lookup without a collection scan: `meds.userId`.

The default migration output is the proposed, sanitized operation list. Apply remains disabled unless both `--apply` and `--confirm-apply` are supplied.
